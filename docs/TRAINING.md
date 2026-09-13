# DeepForest Tree Crown Detection: Training & Fine-Tuning Guide

This guide documents the model fine-tuning and evaluation engine for tree crown detection in **Canopy Core**. The engine enables transfer learning with pretrained DeepForest RetinaNet models on custom regional drone, aerial, and satellite orthomosaics with full Apple Silicon Metal Performance Shaders (MPS) hardware acceleration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Components](#module-components)
3. [Dataset Format & Preparation](#dataset-format--preparation)
4. [Hardware Acceleration (Apple Silicon MPS)](#hardware-acceleration-apple-silicon-mps)
5. [Hyperparameter Optimization Guide](#hyperparameter-optimization-guide)
6. [Evaluation Metrics & Benchmark Standard](#evaluation-metrics--benchmark-standard)
7. [CLI Training & Smoke Test](#cli-training--smoke-test)
8. [Python API Usage](#python-api-usage)
9. [Checkpoint Management & Inference](#checkpoint-management--inference)

---

## 1. Architecture Overview

Canopy Core leverages the **DeepForest RetinaNet** architecture:
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN) extracting multi-scale geospatial features from $P_3$ to $P_7$.
- **Subnets**: Separate classification and regression subnetworks predicting tree crown bounding boxes and confidence scores.
- **Loss Function**: Focal Loss for foreground-background class imbalance and Smooth L1 Loss for box coordinate regression.
- **Pretrained Baseline**: Weights initialized from `weecology/deepforest-tree` (trained on over 30 million tree crowns across diverse forest ecosystems).

```
   Aerial/Drone RGB Tile (e.g. 400x400)
                 │
                 ▼
       ┌──────────────────┐
       │ ResNet-50 + FPN  │ (Multi-Scale Feature Extractor)
       └─────────┬────────┘
                 ├──────────────────────────────┐
                 ▼                              ▼
      ┌────────────────────┐         ┌────────────────────┐
      │ Class Subnet       │         │ Box Reg Subnet     │
      │ (Focal Loss)       │         │ (Smooth L1 Loss)   │
      └──────────┬─────────┘         └──────────┬─────────┘
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
                      Non-Maximum Suppression
                                ▼
                  Predicted Tree Crown Boxes
```

---

## 2. Module Components

- **`canopy_core/models/trainer.py`**:
  - `TrainingConfig`: Type-annotated configuration dataclass controlling data paths, optimization, LR schedules, and hardware settings.
  - `TreeTrainer`: PyTorch Lightning orchestrator managing model setup, MPS device selection, optimizer callbacks, checkpointing, and history logging.
- **`canopy_core/models/evaluator.py`**:
  - `TreeEvaluator`: Evaluation engine performing greedy IoU matching between predictions and ground-truth boxes.
  - `EvaluationResult`: Metrics container computing Precision, Recall, F1 score, and mAP (mAP@0.40, mAP@0.50, mAP@0.75, and mAP@[0.50:0.95] COCO standard).
- **`scripts/run_training.py`**:
  - Full-featured CLI script supporting custom training runs, parameter overrides, post-training evaluation, and a 1-epoch smoke test on `OSBS_029.csv`.

---

## 3. Dataset Format & Preparation

Annotations are supplied as standard comma-separated values (CSV) files with pixel bounding box coordinates.

### CSV Schema

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `image_path` | string | Filename or relative path to the image | `OSBS_029.tif` or `tiles/tile_01.png` |
| `xmin` | float / int | Left coordinate of the bounding box (pixels) | `203` |
| `ymin` | float / int | Top coordinate of the bounding box (pixels) | `67` |
| `xmax` | float / int | Right coordinate of the bounding box (pixels) | `227` |
| `ymax` | float / int | Bottom coordinate of the bounding box (pixels) | `90` |
| `label` | string | Class label identifier | `Tree` |

### Coordinate Constraints
- Origin `(0, 0)` is at the **top-left** of each image tile.
- $0 \le \text{xmin} < \text{xmax} \le \text{image\_width}$
- $0 \le \text{ymin} < \text{ymax} \le \text{image\_height}$
- Box area $(\text{xmax} - \text{xmin}) \times (\text{ymax} - \text{ymin}) > 0$.

### Sample CSV Snippet
```csv
image_path,xmin,ymin,xmax,ymax,label
OSBS_029.tif,203,67,227,90,Tree
OSBS_029.tif,256,99,288,140,Tree
OSBS_029.tif,166,253,225,304,Tree
OSBS_029.tif,365,2,400,27,Tree
OSBS_029.tif,312,13,349,47,Tree
```

### Directory Organization Recommendation
```
data/
├── train/
│   ├── images/
│   │   ├── tile_001.tif
│   │   └── tile_002.tif
│   └── train_annotations.csv
└── val/
    ├── images/
    │   ├── tile_101.tif
    │   └── tile_102.tif
    └── val_annotations.csv
```

---

## 4. Hardware Acceleration (Apple Silicon MPS)

Canopy Core is optimized for Apple Silicon (M1/M2/M3/M4) via PyTorch's Metal Performance Shaders (`mps`) backend.

### Key Considerations for Apple Silicon:
- **Automatic Detection**: When `accelerator="auto"` or `accelerator="mps"`, `resolve_accelerator()` verifies `torch.backends.mps.is_available()` and binds execution to the unified GPU memory.
- **DataLoader Workers**: Set `--num-workers 0` on macOS. In PyTorch with MPS, `num_workers=0` avoids inter-process shared memory contention and memory copy bottlenecks.
- **Precision**: PyTorch MPS supports single precision (`32-true`). Canopy Core defaults to `32-true` for numerical stability in RetinaNet anchor box regression.
- **Unified Memory Efficiency**: M4 unified memory allows zero-copy operations between CPU and GPU, enabling fast 400x400 patch inference and low-latency training epochs.

---

## 5. Hyperparameter Optimization Guide

Fine-tuning object detectors on high-resolution canopy imagery requires balancing feature retention with domain adaptation:

| Hyperparameter | Recommended Range | Default | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **Learning Rate (`lr`)** | `5e-5` – `2e-4` | `1e-4` | Lower LR prevents destabilizing pretrained ResNet backbone weights while allowing detection heads to adapt. |
| **Optimizer** | `AdamW` or `SGD` | `AdamW` | `AdamW` with decoupled weight decay provides superior convergence on small/medium tree datasets compared to vanilla SGD. |
| **Weight Decay** | `1e-5` – `1e-3` | `1e-4` | $L_2$ regularization on convolution kernel weights to suppress overfitting on repetitive canopy textures. |
| **Warmup Epochs** | `1` – `3` | `1` | **Linear warmup** gradually scales LR from `10%` to `100%` during early iterations to avoid large initial gradient spikes from new labels. |
| **LR Scheduler** | `cosine`, `step`, `linear` | `cosine` | **Cosine Annealing** smoothly decays LR to `min_lr` (`1e-6`), providing finer bounding box boundary localization in final epochs. |
| **Batch Size** | `2` – `8` | `4` | Batch size of 4 fits comfortably within Apple Silicon unified memory (16GB–36GB) for 400x400 patches. |
| **Epochs** | `10` – `30` | `5` | Transfer learning converges quickly; 15-20 epochs are typically sufficient for 500-2000 annotated trees. |
| **Patch Size** | `400` – `800` | `400` | Aligns with DeepForest receptive field. Larger patches (e.g. 800) can capture wider context but increase memory consumption. |
| **Patch Overlap** | `0.05` – `0.20` | `0.05` | 5% overlap ensures trees falling on patch boundaries are not truncated during sliding-window inference. |

---

## 6. Evaluation Metrics & Benchmark Standard

Evaluation adheres to object detection standards:

### 1. Intersection over Union (IoU)
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
A prediction is considered a **True Positive (TP)** if:
- $\text{IoU}(P_i, G_j) \ge \text{threshold}$ (default: `0.40` for forestry standard, or `0.50` for PASCAL VOC standard).
- The class label matches.
- The ground-truth box $G_j$ has not already been assigned to a higher-confidence prediction (greedy bipartite matching).

### 2. Precision, Recall & F1 Score
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}$$

### 3. Mean Average Precision (mAP)
Average Precision is calculated using the continuous non-decreasing precision envelope (COCO / PASCAL VOC 2010+):
$$p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$
$$\text{AP} = \sum_{k} (r_{k+1} - r_k) \cdot p_{\text{interp}}(r_{k+1})$$

Canopy Core evaluates four complementary mAP metrics:
- **mAP@0.40**: DeepForest forestry benchmark threshold.
- **mAP@0.50**: PASCAL VOC standard threshold.
- **mAP@0.75**: Strict IoU threshold for precise crown perimeter delineation.
- **mAP@[0.50:0.95]**: COCO standard (average AP across IoU 0.50 to 0.95 in 0.05 steps).

---

## 7. CLI Training & Smoke Test

### One-Command Smoke Test
Verify the entire pipeline on Apple Silicon M4 in ~4 seconds:
```bash
python scripts/run_training.py --smoke-test
```
This runs 1 epoch on the sample `OSBS_029` dataset, tests MPS acceleration, verifies checkpoint writing, and prints evaluation metrics.

### Fine-Tuning on Custom Dataset
```bash
python scripts/run_training.py \
  --train-csv data/train_annotations.csv \
  --train-root-dir data/train_images/ \
  --val-csv data/val_annotations.csv \
  --val-root-dir data/val_images/ \
  --epochs 15 \
  --batch-size 4 \
  --lr 1e-4 \
  --optimizer AdamW \
  --warmup-epochs 2 \
  --lr-scheduler cosine \
  --accelerator mps \
  --checkpoint-dir checkpoints/regional_canopy/ \
  --iou-threshold 0.50
```

### Key CLI Flags
- `--smoke-test`: Quick 1-epoch verification on internal sample data.
- `--train-csv PATH`: Path to annotation CSV.
- `--train-root-dir PATH`: Directory containing image files.
- `--val-csv PATH`: Optional validation annotation CSV.
- `--epochs INT`: Number of training epochs (default: 5).
- `--batch-size INT`: Batch size (default: 4).
- `--lr FLOAT`: Peak learning rate (default: 1e-4).
- `--optimizer {AdamW, SGD}`: Optimizer choice (default: AdamW).
- `--warmup-epochs INT`: Warmup epochs before LR decay (default: 1).
- `--lr-scheduler {cosine, step, linear, constant}`: Decay schedule.
- `--accelerator {auto, mps, cuda, cpu}`: Device backend (default: auto).
- `--iou-threshold FLOAT`: Evaluation IoU threshold (default: 0.5).
- `--skip-eval`: Skip post-training evaluation.

---

## 8. Python API Usage

### Training Programmatically

```python
from canopy_core.models.trainer import TrainingConfig, TreeTrainer

# 1. Configure hyperparameters
config = TrainingConfig(
    train_csv="data/train_annotations.csv",
    train_root_dir="data/train_images",
    val_csv="data/val_annotations.csv",
    val_root_dir="data/val_images",
    epochs=15,
    batch_size=4,
    lr=1e-4,
    optimizer="AdamW",
    warmup_epochs=2,
    lr_scheduler="cosine",
    accelerator="mps",  # Apple Silicon acceleration
    checkpoint_dir="checkpoints/run_01",
)

# 2. Instantiate and train
trainer = TreeTrainer(config)
summary = trainer.fit()

print(f"Training completed in {summary['elapsed_seconds']}s")
print(f"Best checkpoint saved at: {summary['best_checkpoint']}")
print(f"Final PyTorch weights saved at: {summary['final_weights_pt']}")
```

### Evaluating Programmatically

```python
from canopy_core.models.evaluator import TreeEvaluator

evaluator = TreeEvaluator(iou_threshold=0.5, score_threshold=0.15)

# Evaluate predictions DataFrame against ground-truth CSV
results = evaluator.evaluate(
    predictions="data/predictions.csv",
    ground_truth="data/val_annotations.csv",
)

# Print formatted summary table
print(results.summary())

# Access raw metrics
print(f"Precision: {results.precision:.3f}")
print(f"Recall:    {results.recall:.3f}")
print(f"F1 Score:  {results.f1:.3f}")
print(f"mAP@50:    {results.map_50:.3f}")
print(f"COCO mAP:  {results.map_coco:.3f}")

# Export JSON report
results.save_json("checkpoints/run_01/evaluation_report.json")
```

---

## 9. Checkpoint Management & Inference

Canopy Core saves multiple formats for maximum flexibility:

### Generated Checkpoint Files in `checkpoints/`
1. `tree-retinanet-epoch=XX-val_loss=X.XXX.ckpt`: Full PyTorch Lightning checkpoint of the best epoch (contains model weights, optimizer state, scheduler state, and epoch counters).
2. `last.ckpt`: Checkpoint from the final epoch for resuming interrupted runs.
3. `tree_retinanet_final.pt`: Standalone PyTorch `state_dict` for direct, lightweight inference without Lightning dependencies.
4. `training_summary.json`: Record of hyperparameters, training time, loss curves, and artifact paths.
5. `evaluation_report.json`: JSON output of Precision, Recall, F1, and mAP metrics.

### Loading Checkpoints for Inference

```python
from canopy_core.models.trainer import TreeTrainer

# Load from PyTorch Lightning checkpoint (.ckpt) or PyTorch state_dict (.pt)
model = TreeTrainer.load_from_checkpoint(
    checkpoint_path="checkpoints/run_01/tree_retinanet_final.pt"
)

# Run prediction on a new aerial tile
predictions = model.predict_image(path="data/test_images/sample_tile.tif")
print(predictions[["xmin", "ymin", "xmax", "ymax", "score", "label"]].head())
```
