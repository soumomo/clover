#!/usr/bin/env python3
"""
CLI tool for fine-tuning DeepForest tree crown detection models.

Supports:
- Apple Silicon MPS GPU acceleration
- Learning rate warmup & cosine/step decay
- Custom dataset fine-tuning via CSV annotations (image_path, xmin, ymin, xmax, ymax, label)
- Automated checkpoint saving to checkpoints/
- Post-training evaluation with Precision, Recall, F1, and mAP metrics
- One-command smoke test: python scripts/run_training.py --smoke-test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from deepforest import get_data
from canopy_core.models.trainer import TrainingConfig, TreeTrainer, resolve_accelerator
from canopy_core.models.evaluator import TreeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepForest RetinaNet on custom tree crown datasets."
    )

    # Smoke test mode
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 1-epoch smoke test on sample OSBS_029 dataset to verify the pipeline.",
    )

    # Dataset arguments
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Path to training CSV annotations file (image_path, xmin, ymin, xmax, ymax, label).",
    )
    parser.add_argument(
        "--train-root-dir",
        type=str,
        default=None,
        help="Directory containing training images. Defaults to directory of train-csv.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Path to validation CSV annotations file.",
    )
    parser.add_argument(
        "--val-root-dir",
        type=str,
        default=None,
        help="Directory containing validation images. Defaults to directory of val-csv.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Dataloader batch size (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd", "AdamW", "SGD"],
        default="AdamW",
        help="Optimizer type (default: AdamW).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 1e-4).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Number of linear warmup epochs (default: 1).",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["cosine", "step", "linear", "constant"],
        default="cosine",
        help="Learning rate decay scheduler (default: cosine).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at end of decay (default: 1e-6).",
    )

    # Hardware & Performance
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Hardware accelerator (default: 'auto' detects MPS on Apple Silicon).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of compute devices (default: 1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0 for MPS stability).",
    )

    # Checkpoint & Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: 'checkpoints').",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of best checkpoints to retain (default: 1).",
    )

    # Post-training evaluation
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation true positive matching (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine paths based on smoke-test flag
    if args.smoke_test:
        logger.info("=" * 60)
        logger.info("RUNNING 1-EPOCH SMOKE TEST ON OSBS_029 SAMPLE DATASET")
        logger.info("=" * 60)
        sample_csv = get_data("OSBS_029.csv")
        sample_root = os.path.dirname(sample_csv)

        train_csv = sample_csv
        train_root = sample_root
        val_csv = sample_csv
        val_root = sample_root
        epochs = 1
        batch_size = 1
        warmup_epochs = 0
        checkpoint_dir = os.path.join(args.checkpoint_dir, "smoke_test")
    else:
        if not args.train_csv:
            logger.error("Error: --train-csv is required when not in --smoke-test mode.")
            sys.exit(1)
        train_csv = args.train_csv
        train_root = args.train_root_dir
        val_csv = args.val_csv
        val_root = args.val_root_dir
        epochs = args.epochs
        batch_size = args.batch_size
        warmup_epochs = args.warmup_epochs
        checkpoint_dir = args.checkpoint_dir

    # Detect accelerator
    selected_accel = resolve_accelerator(args.accelerator)
    logger.info(f"Target accelerator: {selected_accel.upper()}")

    # Build TrainingConfig
    config = TrainingConfig(
        train_csv=train_csv,
        train_root_dir=train_root,
        val_csv=val_csv,
        val_root_dir=val_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr,
        optimizer=args.optimizer.capitalize() if args.optimizer.lower() == "sgd" else "AdamW",
        weight_decay=args.weight_decay,
        warmup_epochs=warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        min_lr=args.min_lr,
        accelerator=selected_accel,
        devices=args.devices,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        save_top_k=args.save_top_k,
        seed=args.seed,
    )

    logger.info("Initializing TreeTrainer...")
    trainer = TreeTrainer(config)

    logger.info(f"Starting training for {epochs} epoch(s)...")
    summary = trainer.fit()

    logger.info("=" * 60)
    logger.info("TRAINING FINISHED SUCCESSFULLY!")
    logger.info(f"  - Elapsed Time   : {summary['elapsed_seconds']}s")
    logger.info(f"  - Best Checkpoint: {summary['best_checkpoint']}")
    logger.info(f"  - Last Checkpoint: {summary['last_checkpoint']}")
    logger.info(f"  - Final Weights  : {summary['final_weights_pt']}")
    logger.info("=" * 60)

    # Post-training evaluation
    if not args.skip_eval:
        eval_csv = val_csv or train_csv
        eval_root = val_root or train_root
        logger.info(f"Running post-training evaluation at IoU threshold {args.iou_threshold}...")
        eval_res = trainer.evaluate(
            test_csv=eval_csv,
            root_dir=eval_root,
            iou_threshold=args.iou_threshold,
        )
        print("\n" + eval_res.summary() + "\n")

        # Save evaluation report to JSON
        eval_report_path = Path(checkpoint_dir) / "evaluation_report.json"
        eval_res.save_json(eval_report_path)
        logger.info(f"Evaluation report saved to: {eval_report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
