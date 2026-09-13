"""
DeepForest RetinaNet fine-tuning module for tree crown detection.

Features:
- Configures PyTorch Lightning trainer with Apple Silicon MPS GPU acceleration.
- Custom learning rate schedule with Linear warmup and Cosine / Step / Linear decay.
- Batch size, epoch, optimizer (AdamW / SGD), and weight decay controls.
- Checkpoint management: saves best and latest weights to designated checkpoints directory.
- Built-in validation, metrics logging, and evaluation hooks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

import deepforest
from deepforest import main

from canopy_core.models.evaluator import EvaluationResult, TreeEvaluator

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Configuration parameters for DeepForest fine-tuning.
    """
    # Data arguments
    train_csv: Union[str, Path]
    train_root_dir: Optional[Union[str, Path]] = None
    val_csv: Optional[Union[str, Path]] = None
    val_root_dir: Optional[Union[str, Path]] = None

    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-4
    optimizer: str = "AdamW"  # 'AdamW' or 'SGD'
    momentum: float = 0.9     # For SGD
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    warmup_start_factor: float = 0.1
    lr_scheduler: str = "cosine"  # 'cosine', 'linear', 'step', 'constant', 'plateau'
    lr_step_size: int = 3
    lr_decay_gamma: float = 0.1
    min_lr: float = 1e-6

    # Hardware & Performance
    accelerator: str = "auto"  # 'auto', 'mps', 'cuda', 'cpu'
    devices: int = 1
    num_workers: int = 0       # 0 is safest and most performant for MPS dataloaders
    fast_dev_run: bool = False
    precision: str = "32-true"

    # Model architecture & labels
    model_name: str = "weecology/deepforest-tree"
    revision: str = "main"
    num_classes: int = 1
    label_dict: Dict[str, int] = field(default_factory=lambda: {"Tree": 0})
    score_thresh: float = 0.1
    nms_thresh: float = 0.05
    patch_size: int = 400
    patch_overlap: float = 0.05

    # Checkpoint & Logging
    checkpoint_dir: Union[str, Path] = "checkpoints"
    save_top_k: int = 1
    save_last: bool = True
    monitor: Optional[str] = None  # Inferred: 'val_loss' if val_csv else 'train_loss'
    monitor_mode: str = "min"
    log_root: str = "lightning_logs"
    seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        d = asdict(self)
        d["train_csv"] = str(self.train_csv)
        if self.train_root_dir:
            d["train_root_dir"] = str(self.train_root_dir)
        if self.val_csv:
            d["val_csv"] = str(self.val_csv)
        if self.val_root_dir:
            d["val_root_dir"] = str(self.val_root_dir)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        return d


# -----------------------------------------------------------------------------
# Hardware Acceleration Helper
# -----------------------------------------------------------------------------

def resolve_accelerator(preferred: str = "auto") -> str:
    """
    Resolve and validate the optimal hardware accelerator for Apple Silicon / PyTorch.

    Returns:
        str: 'mps', 'cuda', or 'cpu'
    """
    pref = preferred.lower()
    if pref in ["mps", "cuda", "cpu"]:
        if pref == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS accelerator requested but not available. Falling back to CPU.")
            return "cpu"
        if pref == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA accelerator requested but not available. Falling back to CPU.")
            return "cpu"
        return pref

    # Auto-detection
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -----------------------------------------------------------------------------
# Custom Metrics & History Callback
# -----------------------------------------------------------------------------

class MetricsHistoryCallback(Callback):
    """
    PyTorch Lightning callback to collect per-epoch losses and learning rates.
    """
    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        self.history["epoch"].append(current_epoch)

        # Extract logged metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            self.history["train_loss"].append(float(train_loss.item() if hasattr(train_loss, "item") else train_loss))
        else:
            self.history["train_loss"].append(0.0)

        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            self.history["val_loss"].append(float(val_loss.item() if hasattr(val_loss, "item") else val_loss))

        # Extract current learning rate
        try:
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            self.history["lr"].append(float(current_lr))
        except Exception:
            self.history["lr"].append(0.0)

        epoch_str = f"Epoch {current_epoch + 1}/{trainer.max_epochs}"
        loss_str = f"train_loss: {self.history['train_loss'][-1]:.4f}"
        if self.history["val_loss"]:
            loss_str += f", val_loss: {self.history['val_loss'][-1]:.4f}"
        lr_str = f"lr: {self.history['lr'][-1]:.2e}"
        logger.info(f"[{epoch_str}] {loss_str}, {lr_str}")


# -----------------------------------------------------------------------------
# Main Trainer Class
# -----------------------------------------------------------------------------

class TreeTrainer:
    """
    DeepForest RetinaNet fine-tuning orchestrator for custom tree datasets.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration.

        Args:
            config: TrainingConfig object containing training parameters.
        """
        self.config = config
        self._validate_config()

        # Resolve accelerator
        self.accelerator = resolve_accelerator(self.config.accelerator)
        logger.info(
            f"Initialized TreeTrainer: accelerator='{self.accelerator}', "
            f"epochs={self.config.epochs}, batch_size={self.config.batch_size}, "
            f"lr={self.config.lr}, scheduler='{self.config.lr_scheduler}'"
        )

        # Ensure checkpoint and log directories exist
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path(self.config.log_root).mkdir(parents=True, exist_ok=True)

        # Set random seed if specified
        if self.config.seed is not None:
            pl.seed_everything(self.config.seed, workers=True)

        self.model: Optional[main.deepforest] = None
        self.pl_trainer: Optional[pl.Trainer] = None
        self.history_callback: Optional[MetricsHistoryCallback] = None
        self.checkpoint_callback: Optional[ModelCheckpoint] = None

    def _validate_config(self) -> None:
        """Validate input paths and CSV format."""
        train_csv = Path(self.config.train_csv)
        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")

        # Check train CSV headers
        df_head = pd.read_csv(train_csv, nrows=5)
        req_cols = ["image_path", "xmin", "ymin", "xmax", "ymax"]
        missing = [c for c in req_cols if c not in df_head.columns]
        if missing:
            raise ValueError(
                f"Training CSV {train_csv} is missing required bounding box columns: {missing}. "
                f"Expected format: image_path, xmin, ymin, xmax, ymax, label"
            )

        # Infer train root dir if not provided
        if self.config.train_root_dir is None:
            self.config.train_root_dir = train_csv.parent
        else:
            self.config.train_root_dir = Path(self.config.train_root_dir)

        # Validate validation CSV if provided
        if self.config.val_csv is not None:
            val_csv = Path(self.config.val_csv)
            if not val_csv.exists():
                raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
            if self.config.val_root_dir is None:
                self.config.val_root_dir = val_csv.parent
            else:
                self.config.val_root_dir = Path(self.config.val_root_dir)

        # Determine monitor metric
        if self.config.monitor is None:
            self.config.monitor = "val_loss" if self.config.val_csv is not None else "train_loss"

    def _build_optimizer_and_scheduler(self, model: main.deepforest):
        """
        Build optimizer (AdamW or SGD) and LR scheduler with warmup and decay.
        """
        lr = self.config.lr
        weight_decay = self.config.weight_decay
        opt_name = self.config.optimizer.upper()

        if opt_name == "SGD":
            optimizer = torch.optim.SGD(
                model.model.parameters(),
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=weight_decay,
            )
        else:  # AdamW default
            optimizer = torch.optim.AdamW(
                model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        epochs = max(1, self.config.epochs)
        warmup_epochs = max(0, self.config.warmup_epochs)
        sched_type = self.config.lr_scheduler.lower()
        min_lr = self.config.min_lr

        # Case 1: No warmup
        if warmup_epochs <= 0 or warmup_epochs >= epochs:
            if sched_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
            elif sched_type == "step":
                scheduler = StepLR(optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_decay_gamma)
            elif sched_type == "linear":
                end_factor = max(1e-4, min_lr / max(lr, 1e-9))
                scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=epochs)
            elif sched_type == "constant":
                scheduler = ConstantLR(optimizer, factor=1.0, total_iters=epochs)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        else:
            # Case 2: Warmup followed by decay
            decay_epochs = epochs - warmup_epochs
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.config.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            if sched_type == "cosine":
                decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=min_lr)
            elif sched_type == "step":
                decay_scheduler = StepLR(optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_decay_gamma)
            elif sched_type == "linear":
                end_factor = max(1e-4, min_lr / max(lr, 1e-9))
                decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_epochs)
            else:
                decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=min_lr)

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def setup_model(self) -> main.deepforest:
        """
        Initialize and configure DeepForest model with custom settings.
        """
        model = main.deepforest()

        # Update classes and labels if custom mapping provided
        if self.config.num_classes != 1 or (self.config.label_dict and self.config.label_dict != {"Tree": 0}):
            model.config.num_classes = self.config.num_classes
            if self.config.label_dict:
                model.set_labels(self.config.label_dict)

        # Configure hardware & batching in DeepForest config
        model.config.accelerator = self.accelerator
        model.config.devices = self.config.devices
        model.config.workers = self.config.num_workers
        model.config.batch_size = self.config.batch_size
        model.config.score_thresh = self.config.score_thresh
        model.config.nms_thresh = self.config.nms_thresh
        model.config.patch_size = self.config.patch_size
        model.config.patch_overlap = self.config.patch_overlap

        # Dataset configs
        model.config.train.csv_file = str(self.config.train_csv)
        model.config.train.root_dir = str(self.config.train_root_dir)
        model.config.train.epochs = self.config.epochs
        model.config.train.lr = self.config.lr
        model.config.train.fast_dev_run = self.config.fast_dev_run

        if self.config.val_csv is not None:
            model.config.validation.csv_file = str(self.config.val_csv)
            model.config.validation.root_dir = str(self.config.val_root_dir)

        # Initialize deepforest metrics (iou, mAP, precision/recall) if validation data is present
        model.setup_metrics()

        # Hook custom configure_optimizers
        def custom_configure_optimizers():
            return self._build_optimizer_and_scheduler(model)

        model.configure_optimizers = custom_configure_optimizers

        self.model = model
        return model

    def build_trainer(self) -> pl.Trainer:
        """
        Build PyTorch Lightning Trainer configured for Apple Silicon MPS.
        """
        if self.model is None:
            self.setup_model()

        # Logger
        csv_logger = CSVLogger(save_dir=self.config.log_root, name="tree_finetune")

        # Callbacks
        callbacks: List[Callback] = []

        # Metric tracking callback
        self.history_callback = MetricsHistoryCallback()
        callbacks.append(self.history_callback)

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # ModelCheckpoint callback
        monitor_metric = self.config.monitor
        filename = f"tree-retinanet-{{epoch:02d}}-{{{monitor_metric}:.3f}}"
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename=filename,
            monitor=monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k,
            save_last=self.config.save_last,
            verbose=True,
        )
        callbacks.append(self.checkpoint_callback)

        # Configure validation loop limits
        if self.config.val_csv is not None:
            limit_val_batches = 1.0
            num_sanity_val_steps = 1
        else:
            limit_val_batches = 0.0
            num_sanity_val_steps = 0

        # Create Lightning Trainer
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.config.devices,
            max_epochs=self.config.epochs,
            callbacks=callbacks,
            logger=csv_logger,
            fast_dev_run=self.config.fast_dev_run,
            precision=self.config.precision,
            enable_checkpointing=True,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=num_sanity_val_steps,
            default_root_dir=str(self.checkpoint_dir),
            log_every_n_steps=1,
        )

        # Attach trainer to deepforest instance
        self.model.trainer = trainer
        self.pl_trainer = trainer
        return trainer

    def fit(self) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.

        Returns:
            Dict[str, Any]: Training summary, checkpoint paths, and metrics history.
        """
        start_time = time.time()
        logger.info(f"Starting DeepForest fine-tuning on {self.accelerator.upper()}...")

        if self.pl_trainer is None:
            self.build_trainer()

        # Execute training
        self.pl_trainer.fit(self.model)
        elapsed_time = time.time() - start_time

        # Checkpoint paths
        best_model_path = None
        last_model_path = None
        if self.checkpoint_callback is not None:
            best_model_path = self.checkpoint_callback.best_model_path
            last_model_path = self.checkpoint_callback.last_model_path

        # Explicitly save final weights state_dict for standalone PyTorch inference
        final_pt_path = self.checkpoint_dir / "tree_retinanet_final.pt"
        torch.save(self.model.model.state_dict(), str(final_pt_path))
        logger.info(f"Saved final model weights to: {final_pt_path}")

        # Gather history
        history = self.history_callback.history if self.history_callback else {}

        summary = {
            "accelerator": self.accelerator,
            "epochs_completed": self.config.epochs if not self.config.fast_dev_run else 1,
            "elapsed_seconds": round(elapsed_time, 2),
            "best_checkpoint": str(best_model_path) if best_model_path else None,
            "last_checkpoint": str(last_model_path) if last_model_path else None,
            "final_weights_pt": str(final_pt_path),
            "final_train_loss": history.get("train_loss", [-1])[-1] if history.get("train_loss") else None,
            "final_val_loss": history.get("val_loss", [-1])[-1] if history.get("val_loss") else None,
            "config": self.config.to_dict(),
            "history": history,
        }

        # Save summary report to JSON
        summary_json_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved training summary to: {summary_json_path}")

        return summary

    def evaluate(
        self,
        test_csv: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        iou_threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Evaluate the fine-tuned model against test or validation ground truth.

        Args:
            test_csv: Path to evaluation annotations CSV (defaults to config.val_csv)
            root_dir: Image directory (defaults to test_csv parent)
            iou_threshold: Intersection-over-Union threshold for evaluation

        Returns:
            EvaluationResult
        """
        target_csv = test_csv or self.config.val_csv
        if target_csv is None:
            raise ValueError("No evaluation CSV specified and val_csv was not provided in config.")

        evaluator = TreeEvaluator(iou_threshold=iou_threshold, score_threshold=self.config.score_thresh)
        return evaluator.evaluate_model(self.model, test_csv=target_csv, root_dir=root_dir)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model_name: str = "weecology/deepforest-tree",
        num_classes: int = 1,
        label_dict: Optional[Dict[str, int]] = None,
    ) -> main.deepforest:
        """
        Load a trained model from a PyTorch Lightning checkpoint or PyTorch state_dict.

        Args:
            checkpoint_path: Path to .ckpt or .pt file
            model_name: Base architecture
            num_classes: Number of classes
            label_dict: Label dictionary

        Returns:
            deepforest.main.deepforest instance
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        m = main.deepforest()
        if num_classes != 1 or (label_dict and label_dict != {"Tree": 0}):
            m.config.num_classes = num_classes
            if label_dict:
                m.set_labels(label_dict)
        if ckpt_path.suffix == ".pt":
            logger.info(f"Loading state_dict from {ckpt_path}...")
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            m.model.load_state_dict(state_dict)
        elif ckpt_path.suffix == ".ckpt":
            logger.info(f"Loading PyTorch Lightning checkpoint from {ckpt_path}...")
            m = main.deepforest.load_from_checkpoint(str(ckpt_path))
        else:
            raise ValueError(f"Unsupported checkpoint extension: {ckpt_path.suffix}. Expected .pt or .ckpt")

        m.model.eval()
        return m
