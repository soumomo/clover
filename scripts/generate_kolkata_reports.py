import os, time, json
from pathlib import Path
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from canopy_core import TreeDetector, CanopyAreaEngine, SlidingWindowTiler, inspect_raster
from canopy_core.utils.visualization import save_canopy_report_plot

def run_kolkata_inference():
    os.makedirs("reports", exist_ok=True)
    
    samples = [
        {
            "name": "Kolkata Central Park (Banabitan, Salt Lake)",
            "key": "kolkata_central_park",
            "path": "data/samples/kolkata_central_park.tif",
            "score_thresh": 0.28
        },
        {
            "name": "Kolkata Maidan & Victoria Memorial Gardens",
            "key": "kolkata_maidan_victoria",
            "path": "data/samples/kolkata_maidan_victoria.tif",
            "score_thresh": 0.28
        },
        {
            "name": "Sundarbans Biosphere Mangrove Reserve",
            "key": "sundarbans_mangrove",
            "path": "data/samples/sundarbans_mangrove.tif",
            "score_thresh": 0.25
        }
    ]
    
    detector = TreeDetector()
    engine = CanopyAreaEngine()
    summary_results = []
    
    for s in samples:
        print(f"\n=======================================================")
        print(f"  RUNNING INFERENCE: {s['name']}")
        print(f"=======================================================")
        
        t0 = time.time()
        tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)
        boxes_df, meta = detector.predict_tiled_raster(
            s["path"], 
            tiler=tiler, 
            score_threshold=s["score_thresh"]
        )
        elapsed = time.time() - t0
        
        metrics, enriched_df, dissolved_union = engine.compute_metrics(
            boxes_df, 
            raster_metadata=meta
        )
        
        # Read RGB image for visualization
        with rasterio.open(s["path"]) as src:
            rgb_img = src.read([1, 2, 3]).transpose(1, 2, 0)
            transform = src.transform
            
        plot_path = f"reports/{s['key']}_analysis.png"
        save_canopy_report_plot(
            image=rgb_img,
            detections_df=enriched_df,
            metrics=metrics,
            dissolved_geometry=dissolved_union,
            raster_transform=transform,
            output_path=plot_path
        )
        
        # Export crowns to GeoJSON
        if len(enriched_df) > 0:
            crown_ellipses = engine.create_crown_ellipses(enriched_df)
            gdf = gpd.GeoDataFrame(
                enriched_df, 
                geometry=crown_ellipses, 
                crs=meta.crs
            )
            geojson_path = f"reports/{s['key']}_crowns.geojson"
            cols_to_export = [c for c in gdf.columns if c not in ["image_path"]]
            gdf[cols_to_export].to_file(geojson_path, driver="GeoJSON")
            print(f"  [SAVED] Vector GIS Crowns: {geojson_path}")
            
        print(f"  [SAVED] Dual-panel Dashboard: {plot_path}")
        print(f"  Execution Time: {elapsed:.2f}s on Apple M4")
        print(metrics.format_table())
        
        res_dict = metrics.to_dict()
        res_dict["site_name"] = s["name"]
        res_dict["plot_path"] = plot_path
        res_dict["elapsed_seconds"] = round(elapsed, 2)
        summary_results.append(res_dict)
        
    with open("reports/kolkata_inference_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print("\n[ALL DONE] Summary written to reports/kolkata_inference_summary.json")

if __name__ == "__main__":
    run_kolkata_inference()
