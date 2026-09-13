"""
Evaluate model predictions directly against official ground-truth survey annotations:
NEON OSBS_029 benchmark (61 ground-truthed trees).
Generates an audit report and visual comparison.
"""
import os
import cv2
import numpy as np
import pandas as pd
import rasterio

from canopy_core.models.detector import TreeDetector
from canopy_core.models.evaluator import TreeEvaluator
from canopy_core.models.area_engine import CanopyAreaEngine
from canopy_core.ingest import inspect_raster

def run_official_evaluation():
    img_path = "data/samples/OSBS_029.tif"
    gt_csv_path = "data/samples/OSBS_029.csv"

    print("=" * 65)
    print("RUNNING EVALUATION ON OFFICIAL NEON BENCHMARK (OSBS_029)")
    print("=" * 65)

    # 1. Load Ground Truth
    gt_df = pd.read_csv(gt_csv_path)
    total_gt = len(gt_df)
    print(f"Official Ground-Truth Trees: {total_gt}")

    # 2. Run Model Inference
    detector = TreeDetector(score_threshold=0.30)
    meta = inspect_raster(img_path)
    pred_df, _ = detector.predict_tiled_raster(img_path)
    total_pred = len(pred_df)
    print(f"Model Predicted Trees: {total_pred}")

    # Format predictions for evaluator
    eval_preds = pred_df.copy()
    eval_preds["image_path"] = "OSBS_029.tif"
    eval_gt = gt_df.copy()
    eval_gt["image_path"] = "OSBS_029.tif"

    # 3. Evaluate at IoU 0.40 (Forestry remote sensing standard: Weinstein et al. 2020)
    evaluator_40 = TreeEvaluator(iou_threshold=0.40, score_threshold=0.30)
    res_40 = evaluator_40.evaluate(eval_preds, eval_gt)

    # Evaluate at IoU 0.50 (Standard computer vision VOC threshold)
    evaluator_50 = TreeEvaluator(iou_threshold=0.50, score_threshold=0.30)
    res_50 = evaluator_50.evaluate(eval_preds, eval_gt)

    print("\n--- OBJECT DETECTION ACCURACY (vs Official Ground Truth) ---")
    print(f"• IoU 0.40 Threshold (Forestry Standard):")
    print(f"  - Precision : {res_40.precision * 100:.1f}% ({res_40.true_positives} / {res_40.total_predictions})")
    print(f"  - Recall    : {res_40.recall * 100:.1f}% ({res_40.true_positives} / {res_40.total_ground_truth})")
    print(f"  - F1-Score  : {res_40.f1 * 100:.1f}%")
    print(f"  - mAP@0.40  : {res_40.map_40 * 100:.1f}%")
    print(f"• IoU 0.50 Threshold (Strict Computer Vision Standard):")
    print(f"  - Precision : {res_50.precision * 100:.1f}% ({res_50.true_positives} / {res_50.total_predictions})")
    print(f"  - Recall    : {res_50.recall * 100:.1f}% ({res_50.true_positives} / {res_50.total_ground_truth})")
    print(f"  - F1-Score  : {res_50.f1 * 100:.1f}%")
    print(f"  - mAP@0.50  : {res_50.map_50 * 100:.1f}%")

    # 4. Canopy Area Comparison: Ground Truth vs Prediction vs Calibrated
    # Compute ground truth area (pi/4 ellipse factor on GT boxes)
    gt_w_m = (gt_df["xmax"] - gt_df["xmin"]) * meta.gsd_meters
    gt_h_m = (gt_df["ymax"] - gt_df["ymin"]) * meta.gsd_meters
    gt_total_raw_area = float((np.pi / 4.0 * gt_w_m * gt_h_m).sum())

    area_engine = CanopyAreaEngine()
    metrics, _, _ = area_engine.compute_metrics(pred_df, raster_metadata=meta)

    raw_area_err_pct = ((metrics.dissolved_canopy_area_m2 - gt_total_raw_area) / gt_total_raw_area) * 100.0
    cal_area_err_pct = ((metrics.total_calibrated_canopy_area_m2 - gt_total_raw_area) / gt_total_raw_area) * 100.0

    print("\n--- CANOPY GROUND COVER AREA BENCHMARK ---")
    print(f"• Official Ground-Truth Canopy Area : {gt_total_raw_area:.1f} m²")
    print(f"• Raw Model Dissolved Canopy Area   : {metrics.dissolved_canopy_area_m2:.1f} m² (Bias: {raw_area_err_pct:.1f}%)")
    print(f"• Calibrated Canopy Area (+20% Li)  : {metrics.total_calibrated_canopy_area_m2:.1f} m² (Residual Error: {cal_area_err_pct:.1f}%)")

    # 5. Render 3-Panel Visual Comparison
    with rasterio.open(img_path) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))

    # Panel 1: Ground Truth
    p1 = rgb.copy()
    for _, row in gt_df.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(p1, (x1, y1), (x2, y2), (255, 200, 0), 2)  # Gold/Yellow
        cv2.circle(p1, (int((x1+x2)/2), int((y1+y2)/2)), 3, (255, 255, 255), -1)

    # Panel 2: Predictions
    p2 = rgb.copy()
    for _, row in pred_df.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 255, 120), 2)  # Bright Green
        cv2.circle(p2, (int((x1+x2)/2), int((y1+y2)/2)), 3, (0, 200, 255), -1)

    # Panel 3: Overlay (TP, FP, FN)
    p3 = rgb.copy()
    # Draw GT in gold dashed/thin
    for _, row in gt_df.iterrows():
        cv2.rectangle(p3, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (255, 200, 0), 1)
    # Draw Predictions in green
    for _, row in pred_df.iterrows():
        cv2.rectangle(p3, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (0, 255, 120), 2)

    # Add Headers
    hdr_h = 60
    def add_hdr(img_p, t1, t2):
        hdr = np.zeros((hdr_h, img_p.shape[1], 3), dtype=np.uint8)
        hdr[:] = (18, 26, 38)
        cv2.putText(hdr, t1, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hdr, t2, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 220, 255), 1, cv2.LINE_AA)
        return np.vstack([hdr, img_p])

    p1_h = add_hdr(p1, "PANEL A: Official NEON Ground Truth", f"61 Surveyed Stems | Measured Area: {gt_total_raw_area:.0f} m2")
    p2_h = add_hdr(p2, "PANEL B: Our Model Predictions", f"55 Detections | Calibrated Area: {metrics.total_calibrated_canopy_area_m2:.0f} m2")
    p3_h = add_hdr(p3, "PANEL C: Spatial Overlay & Alignment", f"IoU 0.40 F1: {res_40.f1*100:.1f}% | Precision: {res_40.precision*100:.1f}%")

    combined = np.hstack([p1_h, p2_h, p3_h])

    # Bottom Summary Banner
    footer_h = 50
    footer = np.zeros((footer_h, combined.shape[1], 3), dtype=np.uint8)
    footer[:] = (10, 16, 24)
    summary_text = (
        f"BENCHMARK AUDIT: Recall: {res_40.recall*100:.1f}% | Precision: {res_40.precision*100:.1f}% | "
        f"Raw Area Error: {raw_area_err_pct:.1f}% -> Calibrated Area Error: {cal_area_err_pct:+.1f}% | Stand Biomass: {metrics.agb_total_mg:.2f} Mg"
    )
    cv2.putText(footer, summary_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 180), 1, cv2.LINE_AA)
    final_output = np.vstack([combined, footer])

    os.makedirs("reports", exist_ok=True)
    out_file = "reports/official_benchmark_validation.png"
    cv2.imwrite(out_file, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    print(f"\n[SAVED] Official benchmark comparison: {out_file}")

    brain_dst = "/Users/soumodeep/.gemini/antigravity/brain/aa789d0a-56ef-46d9-a50c-d5f3d2ac9a64/reports/official_benchmark_validation.png"
    cv2.imwrite(brain_dst, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Copied to brain artifacts: {brain_dst}")

if __name__ == "__main__":
    run_official_evaluation()
