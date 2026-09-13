"""
Evaluation engine for tree crown detection models.

Computes standard object detection metrics:
- Intersection-over-Union (IoU) box matching
- Precision, Recall, and F1-score (overall and per-class)
- Mean Average Precision (mAP@0.4, mAP@0.5, mAP@0.75, and mAP@[0.5:0.95] COCO style)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Box Geometry & IoU Utilities
# -----------------------------------------------------------------------------

def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU value in [0.0, 1.0]
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between two sets of bounding boxes.

    Args:
        boxes1: (N, 4) array of [xmin, ymin, xmax, ymax]
        boxes2: (M, 4) array of [xmin, ymin, xmax, ymax]

    Returns:
        np.ndarray: (N, M) matrix of pairwise IoU values
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, b1_x2 - b1_x1) * np.maximum(0.0, b1_y2 - b1_y1)
    area2 = np.maximum(0.0, b2_x2 - b2_x1) * np.maximum(0.0, b2_y2 - b2_y1)
    union_area = area1[:, None] + area2[None, :] - inter_area

    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union_area > 0.0, inter_area / union_area, 0.0)

    return ious


# -----------------------------------------------------------------------------
# Precision, Recall & Average Precision Calculations
# -----------------------------------------------------------------------------

def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    method: str = "continuous",
) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve.

    Args:
        recalls: 1D array of recall values (sorted ascending)
        precisions: 1D array of precision values
        method: 'continuous' (VOC 2010+ / COCO standard envelope integration)
                or '11point' (VOC 2007 standard 11-point interpolation)

    Returns:
        float: Average Precision value in [0.0, 1.0]
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    if method == "11point":
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_t = precisions[recalls >= t]
            p = np.max(prec_at_t) if len(prec_at_t) > 0 else 0.0
            ap += p / 11.0
        return float(ap)

    # Continuous AUC integration using monotonically decreasing precision envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return max(0.0, min(1.0, ap))


# -----------------------------------------------------------------------------
# Evaluation Data Structures
# -----------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Complete evaluation metric results container.
    """
    iou_threshold: float
    precision: float
    recall: float
    f1: float
    map_50: float
    map_40: float
    map_75: float
    map_coco: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_ground_truth: int
    total_predictions: int
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    matches_df: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation results to JSON-serializable dictionary."""
        d = asdict(self)
        # Drop matches_df DataFrame from plain dict for serialization
        d.pop("matches_df", None)
        return d

    def summary(self) -> str:
        """Generate a formatted human-readable summary table."""
        lines = [
            "=" * 68,
            f"TREE CROWN DETECTION EVALUATION RESULTS (IoU Threshold: {self.iou_threshold:.2f})",
            "=" * 68,
            f"Overall Precision : {self.precision * 100:6.2f}%",
            f"Overall Recall    : {self.recall * 100:6.2f}%",
            f"Overall F1 Score  : {self.f1 * 100:6.2f}%",
            "-" * 68,
            f"mAP @ IoU 0.40    : {self.map_40 * 100:6.2f}%",
            f"mAP @ IoU 0.50    : {self.map_50 * 100:6.2f}%",
            f"mAP @ IoU 0.75    : {self.map_75 * 100:6.2f}%",
            f"mAP @ [0.50:0.95] : {self.map_coco * 100:6.2f}% (COCO standard)",
            "-" * 68,
            f"Total GT Boxes    : {self.total_ground_truth}",
            f"Total Pred Boxes  : {self.total_predictions}",
            f"True Positives    : {self.true_positives}",
            f"False Positives   : {self.false_positives}",
            f"False Negatives   : {self.false_negatives}",
        ]

        if self.class_metrics:
            lines.append("-" * 68)
            lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP@50':>10} {'Support':>8}")
            lines.append("-" * 68)
            for cls_name, m in self.class_metrics.items():
                p = m.get("precision", 0.0) * 100
                r = m.get("recall", 0.0) * 100
                f = m.get("f1", 0.0) * 100
                ap = m.get("ap_50", 0.0) * 100
                supp = int(m.get("support", 0))
                lines.append(f"{cls_name:<15} {p:>9.2f}% {r:>9.2f}% {f:>9.2f}% {ap:>9.2f}% {supp:>8d}")

        lines.append("=" * 68)
        return "\n".join(lines)

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save evaluation summary metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation metrics to {filepath}")


# -----------------------------------------------------------------------------
# Core Tree Crown Evaluator
# -----------------------------------------------------------------------------

class TreeEvaluator:
    """
    Evaluator for tree crown detection models.

    Matches bounding box predictions against ground-truth annotations across images
    using greedy maximum-IoU bipartite matching and computes precision, recall, F1,
    and mean Average Precision (mAP).
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.1,
        coco_iou_thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            iou_threshold: Primary IoU threshold for TP determination (e.g. 0.4 or 0.5)
            score_threshold: Minimum confidence score to consider for evaluation
            coco_iou_thresholds: List of IoU thresholds for COCO mAP calculation
        """
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        if coco_iou_thresholds is None:
            self.coco_iou_thresholds = [
                float(round(t, 2)) for t in np.arange(0.50, 1.00, 0.05)
            ]
        else:
            self.coco_iou_thresholds = [float(t) for t in coco_iou_thresholds]

    @staticmethod
    def _normalize_df(
        df_or_path: Union[pd.DataFrame, str, Path],
        is_prediction: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize input into standardized DataFrame with columns:
        ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'] (+ 'score' for predictions).
        """
        if isinstance(df_or_path, (str, Path)):
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path.copy()

        # Handle geometry column if coming from GeoPandas
        if "geometry" in df.columns and ("xmin" not in df.columns or df["xmin"].isna().all()):
            bounds = df.geometry.bounds
            df["xmin"] = bounds.minx
            df["ymin"] = bounds.miny
            df["xmax"] = bounds.maxx
            df["ymax"] = bounds.maxy

        req_cols = ["image_path", "xmin", "ymin", "xmax", "ymax"]
        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame is missing required column: '{col}'")

        # Standardize labels
        if "label" not in df.columns:
            df["label"] = "Tree"
        else:
            df["label"] = df["label"].astype(str)

        if is_prediction:
            if "score" not in df.columns:
                df["score"] = 1.0
            else:
                df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1.0)

        # Ensure numeric coordinates
        for col in ["xmin", "ymin", "xmax", "ymax"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Standardize image path to filename
        df["image_path"] = df["image_path"].astype(str).apply(lambda p: Path(p).name)

        # Ensure valid coordinates (xmin < xmax, ymin < ymax)
        x1 = np.minimum(df["xmin"].values, df["xmax"].values)
        x2 = np.maximum(df["xmin"].values, df["xmax"].values)
        y1 = np.minimum(df["ymin"].values, df["ymax"].values)
        y2 = np.maximum(df["ymin"].values, df["ymax"].values)
        df["xmin"], df["xmax"] = x1, x2
        df["ymin"], df["ymax"] = y1, y2

        # Filter zero-area boxes
        valid_area = (df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])
        df = df[valid_area].reset_index(drop=True)

        return df

    def match_image_boxes(
        self,
        pred_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        iou_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """
        Match predictions to ground truth for a single image and compute TP, FP, FN.

        Uses greedy matching sorted by prediction confidence score descending.
        """
        if len(pred_df) == 0 and len(gt_df) == 0:
            return [], 0, 0, 0

        if len(pred_df) == 0:
            # All GT are false negatives
            records = [
                {
                    "gt_idx": idx,
                    "pred_idx": None,
                    "iou": 0.0,
                    "score": 0.0,
                    "match": False,
                    "label": row["label"],
                    "status": "FN",
                }
                for idx, row in gt_df.iterrows()
            ]
            return records, 0, 0, len(gt_df)

        if len(gt_df) == 0:
            # All predictions are false positives
            records = [
                {
                    "gt_idx": None,
                    "pred_idx": idx,
                    "iou": 0.0,
                    "score": row["score"],
                    "match": False,
                    "label": row["label"],
                    "status": "FP",
                }
                for idx, row in pred_df.iterrows()
            ]
            return records, 0, len(pred_df), 0

        # Sort predictions by confidence score descending
        sorted_preds = pred_df.sort_values(by="score", ascending=False).reset_index(drop=True)
        pred_boxes = sorted_preds[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        gt_boxes = gt_df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)

        ious = compute_iou_matrix(pred_boxes, gt_boxes)  # shape (N_preds, M_gt)

        matched_gt = set()
        records: List[Dict[str, Any]] = []
        tp_count = 0
        fp_count = 0

        for p_idx in range(len(sorted_preds)):
            pred_score = float(sorted_preds.loc[p_idx, "score"])
            pred_label = sorted_preds.loc[p_idx, "label"]

            # Candidate ground truth boxes matching class
            candidate_gt = [
                g_idx for g_idx in range(len(gt_df))
                if g_idx not in matched_gt and gt_df.iloc[g_idx]["label"] == pred_label
            ]

            best_iou = 0.0
            best_gt_idx = None

            for g_idx in candidate_gt:
                iou_val = float(ious[p_idx, g_idx])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = g_idx

            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                tp_count += 1
                records.append({
                    "gt_idx": best_gt_idx,
                    "pred_idx": p_idx,
                    "iou": best_iou,
                    "score": pred_score,
                    "match": True,
                    "label": pred_label,
                    "status": "TP",
                })
            else:
                fp_count += 1
                records.append({
                    "gt_idx": best_gt_idx,
                    "pred_idx": p_idx,
                    "iou": best_iou,
                    "score": pred_score,
                    "match": False,
                    "label": pred_label,
                    "status": "FP",
                })

        fn_count = len(gt_df) - len(matched_gt)
        for g_idx in range(len(gt_df)):
            if g_idx not in matched_gt:
                records.append({
                    "gt_idx": g_idx,
                    "pred_idx": None,
                    "iou": 0.0,
                    "score": 0.0,
                    "match": False,
                    "label": gt_df.iloc[g_idx]["label"],
                    "status": "FN",
                })

        return records, tp_count, fp_count, fn_count

    def compute_class_ap_curve(
        self,
        pred_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        label: str,
        iou_threshold: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Average Precision (AP) and precision/recall curve for a specific class.
        """
        class_gt = gt_df[gt_df["label"] == label]
        class_pred = pred_df[pred_df["label"] == label].sort_values(
            by="score", ascending=False
        ).reset_index(drop=True)

        n_gt = len(class_gt)
        if n_gt == 0:
            return (0.0 if len(class_pred) > 0 else 1.0), np.array([]), np.array([])

        if len(class_pred) == 0:
            return 0.0, np.array([]), np.array([])

        # Group ground truth by image
        gt_by_img = {img: g.reset_index(drop=True) for img, g in class_gt.groupby("image_path")}
        matched_gt_per_img = {img: set() for img in gt_by_img}

        tp = np.zeros(len(class_pred), dtype=np.float32)
        fp = np.zeros(len(class_pred), dtype=np.float32)

        for i, row in class_pred.iterrows():
            img = row["image_path"]
            p_box = row[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)

            if img not in gt_by_img:
                fp[i] = 1.0
                continue

            img_gts = gt_by_img[img]
            g_boxes = img_gts[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
            ious = compute_iou_matrix(p_box[None, :], g_boxes)[0]

            best_iou = 0.0
            best_gt_idx = None
            for g_idx in range(len(img_gts)):
                if g_idx not in matched_gt_per_img[img] and ious[g_idx] > best_iou:
                    best_iou = float(ious[g_idx])
                    best_gt_idx = g_idx

            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt_per_img[img].add(best_gt_idx)
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(n_gt)
        precisions = cum_tp / (cum_tp + cum_fp)

        ap = compute_ap(recalls, precisions, method="continuous")
        return ap, recalls, precisions

    def evaluate(
        self,
        predictions: Union[pd.DataFrame, str, Path],
        ground_truth: Union[pd.DataFrame, str, Path],
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground-truth boxes.

        Args:
            predictions: DataFrame or CSV path with predicted boxes
            ground_truth: DataFrame or CSV path with ground-truth boxes

        Returns:
            EvaluationResult: Metrics including precision, recall, F1, and mAP
        """
        pred_df = self._normalize_df(predictions, is_prediction=True)
        gt_df = self._normalize_df(ground_truth, is_prediction=False)

        # Filter by score threshold
        pred_df = pred_df[pred_df["score"] >= self.score_threshold].reset_index(drop=True)

        # Collect unique images across both sets
        all_images = sorted(list(set(pred_df["image_path"]).union(set(gt_df["image_path"]))))
        all_labels = sorted(list(set(pred_df["label"]).union(set(gt_df["label"]))))
        if not all_labels:
            all_labels = ["Tree"]

        all_records: List[Dict[str, Any]] = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Primary IoU threshold matching
        for img_name in all_images:
            img_preds = pred_df[pred_df["image_path"] == img_name].reset_index(drop=True)
            img_gts = gt_df[gt_df["image_path"] == img_name].reset_index(drop=True)

            recs, tp, fp, fn = self.match_image_boxes(
                img_preds, img_gts, iou_threshold=self.iou_threshold
            )
            for r in recs:
                r["image_path"] = img_name
            all_records.extend(recs)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Compute mAP at IoU 0.40, 0.50, 0.75 and COCO mAP@[0.50:0.95]
        def calc_map_at_iou(iou_thresh: float) -> Tuple[float, Dict[str, float]]:
            aps = {}
            for lbl in all_labels:
                ap, _, _ = self.compute_class_ap_curve(pred_df, gt_df, lbl, iou_thresh)
                aps[lbl] = ap
            mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
            return mean_ap, aps

        map_40, _ = calc_map_at_iou(0.40)
        map_50, class_aps_50 = calc_map_at_iou(0.50)
        map_75, _ = calc_map_at_iou(0.75)

        # COCO mAP across thresholds
        coco_aps = []
        for t in self.coco_iou_thresholds:
            m_ap_t, _ = calc_map_at_iou(t)
            coco_aps.append(m_ap_t)
        map_coco = float(np.mean(coco_aps)) if coco_aps else map_50

        # Class-level metrics at primary threshold
        class_metrics: Dict[str, Dict[str, float]] = {}
        for lbl in all_labels:
            c_gt = gt_df[gt_df["label"] == lbl]
            c_preds = pred_df[pred_df["label"] == lbl]
            c_tp = sum(1 for r in all_records if r.get("label") == lbl and r.get("status") == "TP")
            c_fp = sum(1 for r in all_records if r.get("label") == lbl and r.get("status") == "FP")
            c_fn = sum(1 for r in all_records if r.get("label") == lbl and r.get("status") == "FN")

            c_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
            c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = float(2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0

            class_metrics[lbl] = {
                "precision": c_prec,
                "recall": c_rec,
                "f1": c_f1,
                "ap_50": class_aps_50.get(lbl, 0.0),
                "support": float(len(c_gt)),
                "predictions": float(len(c_preds)),
            }

        matches_df = pd.DataFrame(all_records) if all_records else pd.DataFrame()

        return EvaluationResult(
            iou_threshold=self.iou_threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            map_50=map_50,
            map_40=map_40,
            map_75=map_75,
            map_coco=map_coco,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            total_ground_truth=len(gt_df),
            total_predictions=len(pred_df),
            class_metrics=class_metrics,
            matches_df=matches_df,
        )

    def evaluate_model(
        self,
        model: Any,
        test_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """
        Run inference using a DeepForest model and evaluate predictions against ground truth.

        Args:
            model: Trained deepforest.main.deepforest instance
            test_csv: Path to ground-truth annotations CSV
            root_dir: Optional root directory where images are located

        Returns:
            EvaluationResult
        """
        test_csv_path = Path(test_csv)
        if root_dir is None:
            root_dir = test_csv_path.parent
        root_dir = Path(root_dir)

        gt_df = pd.read_csv(test_csv_path)
        image_names = gt_df["image_path"].unique()

        pred_frames = []
        logger.info(f"Evaluating model on {len(image_names)} images from {test_csv_path.name}...")

        for img_name in image_names:
            img_path = root_dir / Path(img_name).name
            if not img_path.exists():
                # Try direct path
                img_path = Path(img_name)
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}, skipping prediction")
                    continue

            try:
                preds = model.predict_image(path=str(img_path))
                if preds is not None and not preds.empty:
                    preds = preds.copy()
                    preds["image_path"] = Path(img_name).name
                    pred_frames.append(preds)
            except Exception as e:
                logger.error(f"Error predicting image {img_path}: {e}")

        if pred_frames:
            all_preds = pd.concat(pred_frames, ignore_index=True)
        else:
            all_preds = pd.DataFrame(columns=["image_path", "xmin", "ymin", "xmax", "ymax", "score", "label"])

        return self.evaluate(predictions=all_preds, ground_truth=gt_df)
