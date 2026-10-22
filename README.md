# Canterbury liquefaction stationarity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23555065.svg)](https://doi.org/10.5281/zenodo.23555065)

Public reproducibility package for the study:

> Non-ergodic but not stationary: repeatable liquefaction manifestation residuals across three Canterbury earthquakes

This release contains the analysis scripts, aggregate outputs, figure exports and
data-source links needed to document the Paper A evidence chain. It deliberately
excludes active manuscript files, cover letters, reviewer material, raw
third-party data, per-sounding source tables and NZGD exports.

## Main result encoded by this release

Using 15,607 CPT case histories at 5,542 Christchurch sites for the 2010
Darfield, 2011 Christchurch and 2016 Christchurch earthquakes, CPT-based
manifestation residuals repeat between consecutive earthquakes (site-to-site
standard deviation 1.47--1.71 logit units for 2010 and 2011 and 1.20--1.31 for
2011 and 2016) but not between 2010 and 2016 (0.00--0.02). A single stationary
site term is rejected: freeing its 2016 loading drives the loading to zero
(deviance falls by 33--53). Observed performance in 2010 and 2011 does not
improve the 2016 ranking beyond the CPT index. An apparently persistent site
term arises only when 122 soundings at ground-improved sites are included.
State dependence (event memory) is not identified: its sign depends on the
assumed site-term distribution and on 36 ground-improved sites.

Companion release: test dates, investigation provenance and depth-limited
ground change, https://doi.org/10.5281/zenodo.23308668.

## Contents

- `src/` -- analysis scripts (Python 3.11; numpy, scipy, pandas, statsmodels,
  matplotlib; the calibration and figure scripts are included)
- `outputs/` -- aggregate JSON and CSV outputs from which the manuscript
  numbers are generated
- `figures/` -- PDF, PNG and SVG exports of the six manuscript figures
- `DATASETS_AND_LINKS.csv` -- source data and access terms
- `REPRODUCIBLE_RUNBOOK.md` -- how the outputs were produced
- `PUBLIC_BOUNDARY.md` -- what is deliberately excluded

## License

MIT; see `LICENSE`. Third-party data remain under their own terms.
