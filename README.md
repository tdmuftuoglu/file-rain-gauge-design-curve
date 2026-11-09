# Rain-Gauge Density and Point-to-Areal Rainfall Representativeness

**How many rain gauges are needed to represent areal rainfall? Design curves against nested, disjoint and true-areal-mean references, with distribution-free tolerance radii, from two dense rain-gauge networks: FIFE / Konza Prairie (Kansas, USA) and the Walnut Gulch Experimental Watershed (Arizona, USA).**

**Author:** Tevfik Denizhan Müftüoğlu

This repository contains a single, self-contained Python script that reproduces every figure, table and number in the accompanying manuscript from the original data files.

## What the script does

1. **Reads and quality-controls both networks.**
   - FIFE: gauge inventory with documented coordinates, conflicting duplicate records, collocated gauges, a spatial neighbour check, a site-to-neighbour consistency check, a comparison of the LTER weighing gauges with the Princeton tipping-bucket gauges during their 1987 overlap (including a day-boundary lag test), and a reconstruction of the daily totals from the 30-minute records.
   - Walnut Gulch: period of record, listed gauge outages and the same neighbour check (the data portal reports missing values as zeros in multi-gauge queries).
2. **Records the daily network composition** and defines a dense set (≥ 20 reporting sites) and a routine set (7–19 sites) for FIFE.
3. **Builds design curves** for the error of a *k*-gauge areal mean against three references: the mean of all gauges (nested), the mean of the remaining gauges (disjoint), and the true areal mean for randomly placed gauges, `RMSE_true(k) = sqrt(mean(CV²)/k)`. Confidence intervals come from a day-cluster bootstrap (percentile and BCa).
4. **Derives distribution-free (conformal) 90 % tolerance radii** and tests their coverage across chronological splits, light and heavy days, seasons, years and network configurations; class-conditional (Mondrian) radii are also computed.
5. **Fits the variance–mean (Taylor) scaling** of spatial variability. A gradient-boosting model is evaluated by repeated 5-fold and leave-one-year-out cross-validation against a power law and a linear model, with out-of-fold SHAP values.
6. **Relates inter-gauge correlation to pairwise error** through the exact decomposition of the mean squared difference.
7. **Compares unweighted and Thiessen (area-weighted) references**, analyses **sub-daily accumulations** (1, 3, 6 and 24 h) from the 30-minute FIFE records, and repeats the design analysis for **Walnut Gulch**: the whole watershed and FIFE-sized 22 km² windows, for the monsoon and non-monsoon seasons.

Output: 13 figures, 34 tables (CSV) and a text summary, bundled into `fife_full_outputs.zip`. Random seeds are fixed, so repeated runs give identical numbers.

## Key results

- **The choice of reference matters.** On dense FIFE days (33 days, 21–30 sites in 22 km²), a single randomly placed gauge has a relative RMSE of 29 % against the true areal mean. With ten gauges the error is 9.2 %, but evaluation against the mean of the same network gives 7.1 %, understating it by nearly a quarter.
- **FIFE design rule.** Three randomly placed gauges give a 20 % RMSE against the true areal mean. Keeping the error within ±20 % on nine days out of ten requires about eight gauges.
- **Climate dominates.** Over the same 22 km², a 20 % RMSE requires 3 gauges at FIFE, 14 at Walnut Gulch outside the monsoon and 38 during the monsoon.
- **Conformal radii.** They held their nominal 90 % coverage under chronological splits (90–95 %), but covered only about 83 % of light days and about 98 % of heavy days. Coverage fell to 83–84 % when the radii were calibrated on one network configuration and applied to another.
- **Variance–mean scaling.** Spatial standard deviation scales as `0.47 · mean^0.62`. Out of sample, the gradient-boosting model did no better than this power law (leave-one-year-out R² 0.61 for both).
- **Correlation is not enough.** Median inter-gauge correlation stays above 0.9 up to 4 km while the pairwise error almost triples. Correlation-based spacing rules therefore hide most of the error growth.

## Data

The script does not redistribute any data; download the files from the sources below.

| Input | Source | File(s) |
|---|---|---|
| FIFE daily rainfall (required) | Huemmrich, K. F., & Briggs, J. (1994). *Daily Rainfall Data (FIFE)*. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/29 | daily archive as downloaded (.rar, .zip or .tar) |
| FIFE 30-minute rainfall (optional; sub-daily analysis) | Wood, E. F. (1994). *30 Minute Rainfall Data (FIFE)*. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2 | 30-minute archive as downloaded |
| Walnut Gulch daily precipitation (optional) | USDA-ARS Southwest Watershed Research Center, Data Access Project, https://www.tucson.ars.ag.gov/dap/ → Precipitation → Digital → Aggregate data: watershed 63, all gauges, time step Daily, units mm, no statistics, output format Excel | `dap_aggregate.xls` |
| Walnut Gulch gauge locations and boundary (optional) | Same site → GIS Data Layers → "Instrumentation" and "Boundary" | `instrumentation.zip`, `boundary.zip` |

A free NASA Earthdata login is required for the ORNL DAAC files. For the Walnut Gulch network, see also Goodrich et al. (2008), *Water Resources Research* 44, W05S04, https://doi.org/10.1029/2006WR005782.

## How to run

**Google Colab (recommended):** paste `fife_representativeness.py` into a single cell (or open `fife_representativeness.ipynb`) and run it. A file picker appears on every run; select all input files at once. The script installs its dependencies, runs the full analysis and downloads `fife_full_outputs.zip`. If only the FIFE daily archive is supplied, the optional parts are skipped.

**Locally:**

```bash
pip install numpy pandas matplotlib scipy pyproj shapely lightgbm shap libarchive-c pyshp lxml
# place the input files next to the script, then:
python fife_representativeness.py
```

## Repository contents

```
fife_representativeness.py      # the complete, single-file analysis
fife_representativeness.ipynb   # the same script as a one-cell notebook
README.md                       # this file
CITATION.cff                    # citation metadata
LICENSE                         # MIT
```

## Citation

If you use this code or its results, please cite itç

A `CITATION.cff` file is included, so GitHub's "Cite this repository" button gives the same reference. The journal reference of the accompanying article will be added here once it is published.

Please also cite the underlying datasets:

> Huemmrich, K. F., & Briggs, J. (1994). *Daily Rainfall Data (FIFE)*. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/29
>
> Wood, E. F. (1994). *30 Minute Rainfall Data (FIFE)*. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2
>
> Goodrich, D. C., Keefer, T. O., Unkrich, C. L., Nichols, M. H., Osborn, H. B., Stone, J. J., & Smith, J. R. (2008). Long-term precipitation database, Walnut Gulch Experimental Watershed, Arizona, United States. *Water Resources Research*, 44, W05S04. https://doi.org/10.1029/2006WR005782

## Acknowledgements

The FIFE rainfall data are distributed by the Oak Ridge National Laboratory Distributed Active Archive Center (ORNL DAAC). Thanks are due to the Konza Prairie LTER staff, the Princeton University group and the FIFE Information System staff, who collected, processed and curated the original observations.

Datasets were provided by the USDA-ARS Southwest Watershed Research Center. Funding for these datasets was provided by the United States Department of Agriculture, Agricultural Research Service.

## License

The code is released under the MIT License (© 2026 Tevfik Denizhan Müftüoğlu). The FIFE data are distributed by NASA / ORNL DAAC and are subject to the EOSDIS Data Use and Citation Guidance. The Walnut Gulch data are distributed by the USDA-ARS Southwest Watershed Research Center and are subject to its data use agreement.
