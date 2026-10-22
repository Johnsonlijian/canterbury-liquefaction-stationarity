# Reproducible runbook

This public repository documents the analysis route without bundling raw
DesignSafe or NZGD inputs. A full rerun therefore requires the user to obtain
the source data directly and place local paths in a private configuration.

## Environment

Python 3.11 with `numpy`, `scipy`, `pandas`, `pyarrow`, `scikit-learn`,
`statsmodels`, `shapely`, `pyproj` and `matplotlib`. A LaTeX installation with
`elsarticle` is needed only for local manuscript compilation; manuscript source
files are intentionally outside this public release.

## Analysis route

1. Obtain PRJ-2937 and NZGD data under their current terms.
2. Decode the CPT test dates and build the case-history table
   (`decode_cpt_dates.py`, `build_case_table.py`).
3. Join the mapped geomorphology and channel-distance covariates
   (`join_geomorphology.py`).
4. Fit the pairwise and three-event random-effects models and their profile
   likelihoods (`state_dependence.py`, `sd_loading.py`, `sd_mixture.py`,
   `sd_allsites.py`).
5. Run the state-dependence checks with discrete site terms, the all-sites
   variant and the free 2016 loading (`sd_mixture.py`, `sd_allsites.py`,
   `sd_robust.py`).
6. Run the forecast designs (`timing_matched.py`, `pairwise_tau.py`) and the
   geomorphology association analysis (`join_geomorphology.py`,
   `geomorph_site_terms.json`).
7. Reproduce the reported numbers with `final_numbers.py` and compare against
   `outputs/`.
8. Regenerate the figures with `make_figures.py` (aggregate inputs only in the
   public route).

The scripts accept project-specific input/output paths. They are provided as
research code, not as a turnkey data downloader.
