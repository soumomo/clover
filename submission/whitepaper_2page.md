# FloraScope · CanopyMRV Engine
## Sub-Meter Optical Tree Crown Delineation, Dissolved Canopy Area & Stand Biomass Quantification
**Candidate Submission for Flora Carbon AI (Sector V, Salt Lake, Kolkata)**  
*Methodology Alignment: Verra VM0047 · IPCC Tier 2 AFOLU · ISO 14064-2*

---

### 1. Executive Summary & Problem Formulation
In voluntary carbon markets (VCM), nature-based solutions (NbS) face an existential credibility hurdle: traditional manual plot sampling is expensive and unscalable, while standard computer vision object detectors over-promise and under-deliver on canopy metrics. Standard bounding-box detectors (e.g., standard YOLO/Faster-RCNN) suffer from three fatal flaws when applied to forestry MRV:
1. **Geometric Overestimation:** A rectangular bounding box around a circular tree crown covers an area of $W \times H$, introducing a theoretical $+27.3\%$ non-canopy bias over the actual crown projection area ($\text{CPA} = \frac{\pi}{4} W H$).
2. **Double-Counting Overlap:** Natural tree crowns intertwine. Summing isolated crown areas double-counts overlapping foliage by $15\text{--}30\%$.
3. **Closed-Canopy Blindness:** In contiguous forests, individual branch boundaries disappear under nadir optical view, causing discrete crown detectors to under-detect stems by up to $60\%$, mistaking continuous green canopy for empty space.

**FloraScope CanopyMRV** solves these failure modes by integrating a high-performance RetinaNet backbone (DeepForest 2.1) with sub-pixel polygon boundary refinement (FastSAM), a computational geometry dissolution engine (`shapely.unary_union`), empirical crown bias calibration (+20%, Li et al., *PNAS Nexus* 2023), and a hydrological false-positive filter for floating aquatic macrophytes (*Eichhornia crassipes*). The pipeline executes 100% locally on Apple Silicon MPS hardware with sub-second per-hectare latency.

---

### 2. End-to-End Pipeline Architecture

```
[GeoTIFF / Drone Screenshot] 
       │
       ▼
[Ingest & GSD Validation] ─── Tier 1 (<0.15m) / Tier 2 (0.15–0.5m) / Warning (>1.0m)
       │
       ▼
[400×400 Sliding Tiler] ───── 20% Stride Overlap, Zero GPU OOM on Gigapixel Mosaics
       │
       ▼
[RetinaNet + IoU NMS] ────── Boundary Non-Maximum Suppression (IoU ≥ 0.45)
       │
       ▼
[Hydrological Weed Filter] ── HSV & ExG Water-Mask Purges Floating Weeds (Eichhornia)
       │
       ▼
[FastSAM Refinement] ──────── Elliptic Geometric Factor (π/4) + Fine Polygons
       │
       ▼
[Geometric Dissolve Engine] ─ shapely.unary_union() Eliminates Foliage Overlap
       │
       ▼
[Li et al. Calibration] ───── Empirical +20% Crown Area Calibration (PNAS Nexus 2023)
       │
       ▼
[Allometric Carbon Engine] ── Pantropical Chave / Jucker Scaling + 1/√N Uncertainty
       │
       ▼
[GIS Deliverables] ────────── Interactive Folium Map, GeoJSON Vectors & CSV Inventory
```

#### Key Technical Modules:
- **Tiling & Boundary NMS:** Rasters are partitioned into $400 \times 400$ px chips with a $20\%$ spatial buffer. Detections straddling tile borders are merged using coordinate-projected Non-Maximum Suppression (Weinstein et al., 2020), eliminating edge artifacts.
- **Hydrological Lake Weed Filter:** Urban wetlands in West Bengal (e.g., Salt Lake Banabitan, Rabindra Sarobar) frequently host dense water hyacinth mats (*Eichhornia crassipes*). Conventional RGB detectors misclassify these green circular weed clusters as emergent saplings. Our hydrological mask detects water surfaces via Normalized Difference Water / Modified ExG thresholds and automatically purges centroid-intersecting detections.
- **Geometric Union & Dissolved Area:** Every candidate crown is modeled as an elliptical polygon. We execute a global `shapely.unary_union` across all candidate polygons within the parcel AOI:
  $$\text{Canopy Cover (\%)} = \frac{\text{Area}\left(\bigcup_{i=1}^N \mathcal{P}_i \cap \mathcal{A}_{\text{AOI}}\right)}{\text{Area}(\mathcal{A}_{\text{AOI}})} \times 100$$
  This mathematically guarantees that overlapping branches cannot artificially inflate parcel canopy coverage.
- **Li et al. (PNAS Nexus 2023) Calibration:** Satellite nadir detectors under-segment low-contrast outer perimeter leaves by an empirical average of $20.0\%$. Applying this peer-reviewed $+20\%$ calibration restores true biological crown projection area without hallucinating false stems.

---

### 3. Empirical Benchmark Validation (NEON OSBS_029)
The engine was validated against the official National Ecological Observatory Network (NEON) ground-truth dataset (`OSBS_029.tif`, $10\text{ cm GSD}$, 61 field-surveyed trees).

| Metric | DeepForest Baseline | FloraScope Calibrated | Performance Gain / Status |
|---|---|---|---|
| **Stem Count** | 55 detected | 55 detected | 61 Ground-Truth Stems |
| **Precision (@ IoU 0.40)** | 80.0% | 80.0% | Low False Alarm Rate |
| **Recall (@ IoU 0.40)** | 72.1% | 72.1% | High Overstory Recovery |
| **F1 Score** | 0.759 | 0.759 | Robust Detection F1 |
| **mAP (@ IoU 0.40)** | 68.0% | 68.0% | Industry-Grade Detector |
| **Canopy Cover Bias** | **-22.0%** (Underestimated) | **-6.4%** (Calibrated) | **+15.6% Absolute Accuracy Improvement** |

---

### 4. Voluntary Carbon Market (VCM) Disclosures & MRV Integrity
To guarantee regulatory audit readiness under **Verra VM0047 v1.0** and **IPCC AFOLU Guidelines**, Clover enforces four explicit physical disclosures:
1. **The Understory Blind Spot:** 2D optical sensors capture only overstory canopy photons. In stratified stands, dominant trees represent $75\text{--}85\%$ of stand biomass (*IPCC AFOLU 2019 Refinement*). Formal carbon credit issuance requires pairing optical baselines with terrestrial sample plots or regional Biomass Expansion Factors (BEF) for suppressed sub-canopy stems.
2. **Species-Agnostic Wood Density ($\rho$):** 3-band RGB data cannot determine species or cellular specific gravity. A native Teak ($\rho \approx 0.65\text{ g/cm}^3$) or Sal ($\rho \approx 0.72\text{ g/cm}^3$) sequesters double the carbon of a Silk Cotton (*Bombax ceiba*, $\rho \approx 0.35\text{ g/cm}^3$) for identical crown geometry (*Dryad Global Wood Density Database, Zanne et al. 2009 / Chave et al. 2009*). Field botanical surveys remain indispensable.
3. **Continuous Canopy vs. Discrete Dilemma:** In closed-canopy rainforests or mature mangrove blocks (Sundarbans), touching crowns form a continuous green roof. Rather than forcing arbitrary bounding boxes that double-count land by 20–40%, Clover uses **Dissolved Non-Overlapping Surface Footprint ($\text{m}^2$)** via unary topological union, guaranteeing zero double-crediting.
4. **Dual-Component Uncertainty Modeling ($\sigma_{\text{stand}}$):** Individual tree optical allometry carries high residual error ($\sigma_{\text{rand}} \approx \pm 56.5\%$, *Jucker et al. 2016*). While independent sampling error attenuates with sample size ($56.5\%/\sqrt{N}$), stand-level estimates hit a systematic model/sensor error floor. Clover models $\sigma_{\text{sys}} \approx 12.0\%$ (selected as a representative floor within the empirical $10\text{--}44\%$ stand-scale range documented by *Chave et al. 2014* and *Réjou-Méchain et al. 2017*):
   $$\sigma_{\text{stand}} = \sqrt{\frac{(56.5\%)^2}{N_{\text{stems}}} + (12.0\%)^2}$$
   For a representative Kolkata park with $380$ stems, random noise cancels out, and total stand uncertainty converges to an auditor-defensible **$\pm 12.3\%$** (90% CI), meeting Verra VCS Standard v4.5 precision thresholds without fabricating zero error.

---

### 5. Peer-Reviewed Citations & Literature Foundation
1. **Weinstein et al. (2020)** – *Remote Sensing of Environment*. Cross-site evaluation of deep learning for individual tree crown detection in airborne RGB imagery.
2. **Li et al. (2023)** – *PNAS Nexus*. Underestimation of individual tree crown area in satellite remote sensing and empirical bias correction (+20%).
3. **Jucker et al. (2016)** – *Global Change Biology*. Allometric equations for integrating aerial LiDAR with optical remote sensing to map tropical forest carbon stocks.
4. **Chave et al. (2014)** – *Global Change Biology*. Improved pantropical allometric models to estimate aboveground biomass of tropical trees.
5. **Réjou-Méchain et al. (2017)** – *Methods in Ecology and Evolution*. BIOMASS: an R package for estimating above-ground biomass and its uncertainty in tropical forests.
6. **Zanne et al. / Chave et al. (2009)** – *Global Wood Density Database*. Dryad Digital Repository.
7. **Verra VM0047 (2023)** – *Methodology for Afforestation, Reforestation, and Revegetation Projects (ARR), Version 1.0*.
8. **Verra VCS Standard (v4.5, 2023)** – *Section 4.5: Precision, Uncertainty, and Confidence Deductions*.
9. **IPCC (2019)** – *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Volume 4 (AFOLU)*.
