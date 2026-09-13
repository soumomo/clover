# Clover · Sub-Meter Optical Canopy Delineation & Carbon MRV

Clover is an auditable, scientifically calibrated remote sensing and ecological AI platform for individual tree crown delineation, dissolved topological canopy accounting, and transparent pantropical carbon stock MRV.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tryclover.streamlit.app)
[![DeepForest](https://img.shields.io/badge/DeepForest-2.1-22C55E.svg)](https://deepforest.readthedocs.io/)
[![Standards](https://img.shields.io/badge/Standards-Verra%20VM0047%20%7C%20IPCC%20Tier%202-166534.svg)](https://verra.org/methodologies/vm0047-afforestation-reforestation-and-revegetation-v1-0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

---

## Key Scientific & Engineering Innovations

### 1. Dissolved Non-Overlapping Canopy Footprint
- **The Problem:** In contiguous tropical forests, mangroves, and dense urban parks, tree crowns intertwine into a continuous canopy. Summing naive bounding boxes double-counts overlapping foliage by 20% to 40%, inflating credited carbon.
- **The Solution:** Clover dissolves all individual crown geometries into a continuous topological polygon using geometric unary union (`shapely.unary_union`), guaranteeing that **no square meter of land is double-credited**.

### 2. Empirical Crown Area Bias Calibration (+20%)
- Calibrated against **Li et al. (*PNAS Nexus*, 2023)** empirical findings: optical bounding-box detectors overestimate crown area by +20% in open trees and underestimate in closed canopies. Clover applies bias correction prior to allometric scaling.

### 3. Dual-Component Stand Uncertainty Decomposition
- **The Problem:** Pantropical optical allometry linking 2D Crown Diameter ($CD$) to Aboveground Biomass ($AGB$) carries high single-tree residual error ($\sigma_{\text{rand}} \approx \pm 56.5\%$, *Jucker et al. 2016*). While random sampling error attenuates across $N$ stems ($56.5\%/\sqrt{N}$), stand-level estimates hit an unavoidable systematic model and sensor error floor.
- **The Solution:** Clover models both components honestly rather than claiming unrealistically tiny error:
  $$\sigma_{\text{stand}} = \sqrt{\frac{(56.5\%)^2}{N_{\text{stems}}} + \sigma_{\text{sys}}^2}$$
  where $\sigma_{\text{sys}} \approx 12.0\%$ is selected within the empirical 10–44% stand-scale range documented by *Chave et al. (2014)* and *Réjou-Méchain et al. (2017)*. Total stand uncertainty converges realistically to **$\pm 12.5\%\text{--}13.5\%$ (90% CI)**, compliant with Verra VCS Standard v4.5 precision guidelines.

### 4. Transparent Regulatory Disclosures
- Embedded scientific disclosure popups detailing the **Understory Blind Spot** (optical sensors capture overstory; formal issuance requires ground inventory plots for sub-canopy stems), **Wood Density ($\rho$) Allometric Variance** (*Global Wood Density Database, Dryad; Zanne et al. 2009 / Chave et al. 2009*), and **Additionality / Double-Counting Protection** (*Verra VM0047 v1.0*).

### 5. Multi-Scale Adaptive Feature Pyramid
- Native support for arbitrary resolution inputs: auto-downsamples high-resolution gigapixel drone rasters to optimal effective receptive field (~0.25 m/pixel) while applying multi-scale feature pyramids (1.0x + 2.0x) to resolve both emergent emergent crowns and small saplings.

### 6. Hydrological Lake Weed Rejection Filter
- Geospatially purges floating aquatic weeds (*Eichhornia crassipes* / water hyacinth) inside water bodies, preventing false positives in urban lakes and wetland reserves.

### 7. Auditor-Grade Export Formats
- One-click exports for:
  - **GeoJSON**: Full geospatial polygon contours with individual stem IDs, crown diameters, projected areas, and estimated biomass.
  - **CSV**: Complete tree stem inventory ledger.
  - **Annotated Boxed PNG**: Full-resolution raster with high-contrast emerald bounding boxes (`#22C55E`) and apical center points (`#4ADE80`).

---

## How to Run Clover

You have two simple ways to use Clover:

### Option 1: Live Cloud Application (Instant — 0 Setup)
The official version of Clover is hosted 24/7 on Streamlit Community Cloud:

👉 **[https://tryclover.streamlit.app](https://tryclover.streamlit.app)**

* Zero installation required.
* High-resolution presets, custom GeoTIFF upload, interactive satellite maps, and instant CSV/GeoJSON exports.

---

### Option 2: Run Locally on Your Machine (Powered by `uv`)
For offline field deployments, local GPU acceleration, or custom drone mapping pipelines, run Clover locally using [**`uv`**](https://github.com/astral-sh/uv), the extremely fast Python package manager.

#### 1. Install `uv` (if not already installed)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

#### 2. Clone & Run in One Command
```bash
# Clone the repository
git clone https://github.com/soumomo/clover.git
cd clover

# Run directly (uv automatically creates an isolated environment & runs Streamlit)
uv run streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> [!TIP]
> Alternatively, synchronize your project dependencies deterministically using the committed `uv.lock`:
> ```bash
> uv sync
> uv run streamlit run app.py
> ```

---

## Peer-Reviewed Citations

1. **Weinstein, B. G., et al. (2020)**. Cross-site evaluation of deep learning for individual tree crown detection in airborne RGB imagery. *Remote Sensing of Environment*, 241, 111815. [DOI: 10.1016/j.rse.2020.111815](https://doi.org/10.1016/j.rse.2020.111815)
2. **Li, D., et al. (2023)**. Underestimation of individual tree crown area in satellite remote sensing and empirical bias correction. *PNAS Nexus*, 2(3), pgad098. [DOI: 10.1093/pnasnexus/pgad098](https://doi.org/10.1093/pnasnexus/pgad098)
3. **Jucker, T., et al. (2016)**. Allometric equations for integrating aerial LiDAR with optical remote sensing to map tropical forest carbon stocks. *Global Change Biology*, 23(1), 190–205. [DOI: 10.1111/gcb.13388](https://doi.org/10.1111/gcb.13388)
4. **Chave, J., et al. (2014)**. Improved pantropical allometric models to estimate aboveground biomass in tropical forests. *Global Change Biology*, 20(10), 3177–3190. [DOI: 10.1111/gcb.12629](https://doi.org/10.1111/gcb.12629)
5. **Réjou-Méchain, M., et al. (2017)**. BIOMASS: an R package for estimating above-ground biomass and its uncertainty in tropical forests. *Methods in Ecology and Evolution*, 8(9), 1163–1167. [DOI: 10.1111/2041-210X.12753](https://doi.org/10.1111/2041-210X.12753)
6. **Zanne, A. E., et al. / Chave, J. (2009)**. Global wood density database. *Dryad Digital Repository*. [DOI: 10.5061/dryad.234](https://doi.org/10.5061/dryad.234)
7. **Verra VM0047 (2023)**. Methodology for Afforestation, Reforestation, and Revegetation Projects (ARR), Version 1.0. [Official Standard](https://verra.org/methodologies/vm0047-afforestation-reforestation-and-revegetation-v1-0/)
8. **Verra VCS Standard (v4.5, 2023)**. Precision, Uncertainty, and Confidence Deductions (§4.5).
9. **IPCC (2019)**. 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Volume 4 (AFOLU).

---

## License
MIT License.
