# Rain-Gauge Density and Point-to-Areal Rainfall Representativeness

**A dense-network design curve with distribution-free tolerance guarantees, from the FIFE / Konza daily rain-gauge network.**

This repository contains a single, self-contained Python script that reproduces every figure, table, and number in the accompanying manuscript. It asks a simple, practical question — *how many rain gauges does a small area need to estimate its average rainfall?* — and answers it with a calibrated design curve, distribution-free guarantees, and an explanation of when a single gauge fails.

## What it does

Starting from the raw FIFE Daily Rainfall archive, the script:

1. **Parses and georeferences** 89 data granules (10,034 daily records) into a gauge-by-day matrix, decoding the FIS site-grid identifiers into UTM/geographic coordinates (verified against the dataset metadata).
2. Builds a **gauge-density design curve**: how the error of a *k*-gauge areal estimate falls as *k* increases, with day-cluster bootstrap confidence bands.
3. Derives **distribution-free tolerance radii** with split conformal prediction, and the minimum number of gauges for a target areal accuracy at 90% confidence.
4. Fits the **variance–mean (Taylor) scaling** of spatial rainfall variability and identifies its drivers with a gradient-boosted model + **SHAP**.
5. Contrasts **inter-gauge correlation with pairwise error** (a caution for correlation-based network design).
6. Checks **robustness** to the choice of areal reference (unweighted vs Thiessen area-weighted).

Output: 11 publication-quality figures + 11 tables/CSVs + a text summary, bundled into `fife_full_outputs.zip`.

## Key results

- A single gauge represents the 22 km² areal mean with ~44 % relative RMSE; this falls to ~12 % for 5 gauges and ~4 % for 8.
- At 90 % confidence (distribution-free): **4 gauges** keep areal error within ±30 %, **5 gauges** within ±20 %.
- Spatial standard deviation scales as `SD ∝ mean^0.63`, so light and convective storms are the least well represented.
- Inter-gauge correlation stays above 0.9 across the whole network while pairwise error triples with separation.
- Results are essentially unchanged under an area-weighted (Thiessen) reference (r = 0.999).

## Data

FIFE Daily Rainfall Data Set (Version 1), Konza Prairie, Kansas, USA
Huemmrich, K. F., & Briggs, J. M. (1994). ORNL Distributed Active Archive Center.
DOI: [10.3334/ORNLDAAC/29](https://doi.org/10.3334/ORNLDAAC/29)

The dataset is openly available from the ORNL DAAC (a free NASA Earthdata login is required to download the raw files). The script does **not** redistribute the data; download the archive yourself and point the script at it.

## How to run

**Google Colab (recommended):** paste `fife_representativeness.py` into a single cell and run it. A file-picker appears — select the downloaded `.rar` archive — and the script installs its dependencies, runs the full analysis, and downloads `fife_full_outputs.zip`.

**Locally:**

```bash
pip install numpy pandas matplotlib scipy pyproj shapely lightgbm shap libarchive-c
# place the FIFE .rar next to the script, then:
python fife_representativeness.py
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `pyproj`, `shapely`, `lightgbm`, `shap`, `libarchive-c`.

## Repository contents

```
fife_representativeness.py     # the complete, single-file analysis
README.md                      # this file
LICENSE                        # MIT
```

## Citation

If you use this code, please cite the manuscript (details to follow) and the underlying dataset:

> Huemmrich, K. F., & Briggs, J. M. (1994). *Daily Rainfall Data (FIFE)*, Version 1. ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/29

## License

Code released under the MIT License. The FIFE rainfall data are the property of NASA / ORNL DAAC and are subject to the EOSDIS Data Use and Citation Guidance.
