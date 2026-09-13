"""
Unit and integration tests for Canopy Core Training & Evaluation Engine.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from deepforest import get_data

from canopy_core.models.evaluator import (
    TreeEvaluator,
    EvaluationResult,
    compute_box_iou,
    compute_iou_matrix,
    compute_ap,
)
from canopy_core.models.trainer import (
    TrainingConfig,
    TreeTrainer,
    resolve_accelerator,
)


def test_box_iou():
    """Verify single box IoU computation."""
    b1 = np.array([0, 0, 10, 10])
    b2 = np.array([0, 0, 10, 10])
    assert compute_box_iou(b1, b2) == 1.0

    # Non-overlapping
    b3 = np.array([20, 20, 30, 30])
    assert compute_box_iou(b1, b3) == 0.0

    # Half overlap: 5x10 / (100 + 100 - 50) = 50 / 150 = 1/3
    b4 = np.array([5, 0, 15, 10])
    assert round(compute_box_iou(b1, b4), 4) == round(1.0 / 3.0, 4)


def test_iou_matrix():
    """Verify pairwise vectorized IoU matrix."""
    boxes1 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=float)
    boxes2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)

    ious = compute_iou_matrix(boxes1, boxes2)
    assert ious.shape == (2, 2)
    assert ious[0, 0] == 1.0
    assert ious[1, 0] == 0.0
    assert round(float(ious[0, 1]), 4) == round(25.0 / 175.0, 4)


def test_compute_ap():
    """Verify AP calculation."""
    # Perfect detector
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert compute_ap(recalls, precisions) == 1.0

    # Empty
    assert compute_ap(np.array([]), np.array([])) == 0.0


def test_evaluator_synthetic():
    """Test TreeEvaluator on synthetic ground truth and predictions."""
    gt = pd.DataFrame({
        "image_path": ["tile1.tif", "tile1.tif", "tile2.tif"],
        "xmin": [10, 50, 100],
        "ymin": [10, 50, 100],
        "xmax": [30, 70, 120],
        "ymax": [30, 70, 120],
        "label": ["Tree", "Tree", "Tree"],
    })

    # Predictions: 2 correct matches, 1 false positive
    preds = pd.DataFrame({
        "image_path": ["tile1.tif", "tile1.tif", "tile1.tif"],
        "xmin": [10, 52, 200],
        "ymin": [10, 52, 200],
        "xmax": [30, 72, 220],
        "ymax": [30, 72, 220],
        "score": [0.95, 0.88, 0.50],
        "label": ["Tree", "Tree", "Tree"],
    })

    evaluator = TreeEvaluator(iou_threshold=0.5)
    res = evaluator.evaluate(preds, gt)

    assert res.total_ground_truth == 3
    assert res.total_predictions == 3
    assert res.true_positives == 2
    assert res.false_positives == 1
    assert res.false_negatives == 1
    assert round(res.precision, 4) == round(2.0 / 3.0, 4)
    assert round(res.recall, 4) == round(2.0 / 3.0, 4)
    assert round(res.f1, 4) == round(2.0 / 3.0, 4)
    assert "Tree" in res.class_metrics
    summary = res.summary()
    assert "TREE CROWN DETECTION EVALUATION RESULTS" in summary


def test_accelerator_detection():
    """Verify MPS hardware accelerator detection."""
    accel = resolve_accelerator("auto")
    assert accel in ["mps", "cuda", "cpu"]


def test_training_config():
    """Verify TrainingConfig validation and defaults."""
    csv_file = get_data("OSBS_029.csv")
    cfg = TrainingConfig(
        train_csv=csv_file,
        epochs=2,
        batch_size=2,
        lr=1e-4,
    )
    assert cfg.epochs == 2
    assert cfg.batch_size == 2
    assert cfg.optimizer == "AdamW"
    assert cfg.warmup_epochs == 1
    d = cfg.to_dict()
    assert d["epochs"] == 2
    assert "train_csv" in d


def test_checkpoint_load():
    """Verify loading weights state_dict with TreeTrainer.load_from_checkpoint."""
    ckpt_path = "checkpoints/smoke_test/tree_retinanet_final.pt"
    if os.path.exists(ckpt_path):
        model = TreeTrainer.load_from_checkpoint(ckpt_path)
        assert model is not None
        assert hasattr(model, "predict_image")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
