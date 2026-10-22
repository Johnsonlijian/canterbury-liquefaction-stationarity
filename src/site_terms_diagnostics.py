"""Diagnostics for repeatable site terms.

D1  2016 transfer failure: information content of training-event site terms for the
    held-out 2016 event when the index is weighted by an oracle in-event slope
    (diagnostic only; not a forecast) and for a grid of site-term weights lambda.
D2  Spatial structure: empirical semivariogram and nearest-neighbour correlation of the
    empirical-Bayes site terms (all events, LSN model, epoch-split 2010 equation).
D3  Correlates: site terms versus water-table depth, crust thickness, soil-behaviour
    descriptors, demand uncertainty (PGA sigma) and position; OLS with 1-km clusters.
D4  Pair-wise residual correlation between events: correlation of event-specific
    Pearson residuals (index-only logistic per event) across sites, to see which event
    pairs share repeatable components.

Usage: python -I site_terms_diagnostics.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_terms_glmm import fit, eb_from_params  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_parquet(out / "cases.parquet")
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    c["block"] = (np.floor(c["E"] / 1000).astype(int) * 100000 + np.floor(c["N"] / 1000).astype(int))
    res = {}

    # D1
    d1 = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        c["x"] = np.log1p(c[ix])
        o, eb, theta, names = fit(c, ["2010", "2011"], epoch_split_2010=True)
        ebh = eb_from_params(c[c["event"].isin(["2010", "2011"])], theta, names)
        d = c[c["event"] == "2016"].merge(ebh, on="rec_index")
        m = sm.Logit(d["y"].values, sm.add_constant(d["x"].values)).fit(disp=0)
        b16 = float(m.params[1])
        row = {"b_2016_in_event": b16, "b_train": o["b"], "AUC_index": auc(d["y"], d["x"])}
        for lam in [0.0, 0.1, 0.25, 0.5, 1.0]:
            row[f"AUC_oracle_slope_lambda_{lam}"] = auc(d["y"], b16 * d["x"] + lam * d["d_hat"])
        # in-event logistic: is d_hat significant given x?
        m2 = sm.Logit(d["y"].values, sm.add_constant(np.column_stack([d["x"].values, d["d_hat"].values]))).fit(disp=0)
        row["in_event_coef_dhat"] = [float(m2.params[2]), float(m2.bse[2]), float(m2.pvalues[2])]
        d1[ix] = row
    res["D1_2016_transfer"] = d1

    # D2 spatial structure of EB site terms (LSN, all events)
    eb = pd.read_parquet(out / "site_terms_eb_LSN.parquet")
    snd = pd.read_parquet(out / "soundings.parquet")
    eb = eb.merge(snd[["rec_index", "E", "N", "date"]], on="rec_index")
    xy = eb[["E", "N"]].values
    z = eb["d_mean"].values
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=3000.0, output_type="ndarray")
    dist = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    sq = 0.5 * (z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2
    bins = np.array([0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000])
    vg = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (dist > lo) & (dist <= hi)
        if mk.sum() > 30:
            vg.append({"lo_m": float(lo), "hi_m": float(hi), "n_pairs": int(mk.sum()), "semivariance": float(sq[mk].mean())})
    res["D2_variogram_LSN"] = {"variance_dhat": float(np.var(z)), "bins": vg}
    dist1, idx1 = tree.query(xy, k=2)
    nn = idx1[:, 1]
    for r in [50, 100, 250, 500]:
        mk = dist1[:, 1] <= r
        res["D2_variogram_LSN"][f"nn_corr_le_{r}m"] = [float(np.corrcoef(z[mk], z[nn[mk]])[0, 1]), int(mk.sum())]

    # D3 correlates (descriptors from 2011 rows)
    c11 = pd.read_parquet(out / "cases.parquet")
    c11 = c11[(c11["variant"] == "inv") & (c11["event"] == "2011")][
        ["rec_index", "GWT", "H1", "frac_sandlike_0_10", "median_Ic_0_10", "median_q_liq_0_10", "n_ic_transitions_0_10",
         "PGA", "PGAsig", "crit_depth", "thick_trig_10m"]]
    D = eb.merge(c11, on="rec_index").dropna()
    D["E_km"] = (D["E"] - D["E"].mean()) / 1000.0
    D["N_km"] = (D["N"] - D["N"].mean()) / 1000.0
    D["block"] = (np.floor(D["E"] / 1000).astype(int) * 100000 + np.floor(D["N"] / 1000).astype(int))
    for col in ["GWT", "H1", "frac_sandlike_0_10", "median_Ic_0_10", "median_q_liq_0_10", "n_ic_transitions_0_10", "PGAsig", "thick_trig_10m"]:
        D[f"z_{col}"] = (D[col] - D[col].mean()) / D[col].std()
    f = ("d_mean ~ z_GWT + z_H1 + z_frac_sandlike_0_10 + z_median_Ic_0_10 + z_median_q_liq_0_10 + "
         "z_n_ic_transitions_0_10 + z_PGAsig + z_thick_trig_10m")
    m = smf.ols(f, data=D).fit(cov_type="cluster", cov_kwds={"groups": D["block"]})
    res["D3_correlates_no_position"] = {k: [float(m.params[k]), float(m.bse[k]), float(m.pvalues[k])] for k in m.params.index}
    res["D3_R2_no_position"] = float(m.rsquared)
    m2 = smf.ols(f + " + E_km + N_km + I(E_km**2) + I(N_km**2) + E_km:N_km", data=D).fit(cov_type="cluster", cov_kwds={"groups": D["block"]})
    res["D3_R2_with_position"] = float(m2.rsquared)
    res["D3_abs_dhat_vs_PGAsig_spearman"] = float(pd.Series(np.abs(D["d_mean"])).corr(D["PGAsig"], method="spearman"))

    # D4 pairwise residual correlations (index-only logistic per event, Pearson residuals)
    resid = {}
    for ev in ["2010", "2011", "2016"]:
        d = c[c["event"] == ev].copy()
        d["x"] = np.log1p(d["LSN"])
        if ev == "2010":
            parts = []
            for ep, g in d.groupby("p0"):
                mm = sm.Logit(g["y"].values, sm.add_constant(g["x"].values)).fit(disp=0)
                p = mm.predict(sm.add_constant(g["x"].values))
                parts.append(pd.Series((g["y"].values - p) / np.sqrt(p * (1 - p)), index=g["rec_index"].values))
            resid[ev] = pd.concat(parts)
        else:
            mm = sm.Logit(d["y"].values, sm.add_constant(d["x"].values)).fit(disp=0)
            p = mm.predict(sm.add_constant(d["x"].values))
            resid[ev] = pd.Series((d["y"].values - p) / np.sqrt(p * (1 - p)), index=d["rec_index"].values)
    R = pd.DataFrame(resid)
    res["D4_residual_corr_spearman"] = R.corr(method="spearman").to_dict()
    p0_sites = c[(c["event"] == "2010") & (c["p0"] == 1)]["rec_index"].unique()
    res["D4_residual_corr_spearman_P0_sites"] = R.loc[R.index.isin(p0_sites)].corr(method="spearman").to_dict()
    res["D4_residual_corr_spearman_later_sites"] = R.loc[~R.index.isin(p0_sites)].corr(method="spearman").to_dict()

    (out / "site_terms_diagnostics.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main(sys.argv[1])
