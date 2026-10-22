"""Is the apparent post-earthquake 'recovery' of CPT resistance a sampling artefact?

Metrics per sounding (event-independent soil descriptors in 1-10 m below the 2011 water
table): median qc1Ncs of sand-like layers (Ic < 2.6) and median Ic in 0-10 m.
Time: days since the most recent major Canterbury event (4 Sep 2010, 22 Feb 2011,
13 Jun 2011, 23 Dec 2011) at the test date.

Estimates the slope of each metric on log10(days since last event):
  (1) pooled across all soundings (cross-sectional; what an archive-wide trend shows),
  (2) with 500 m and 250 m grid-cell fixed effects (same neighbourhood, different dates),
  (3) on pairs of soundings <= 20 m apart tested on different dates (first differences).
Cluster-robust SEs by 1 km block.

Usage: python -I apparent_recovery.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

EVENTS = [pd.Timestamp(x) for x in ["2010-09-04", "2011-02-22", "2011-06-13", "2011-12-23"]]


def days_since_last(d):
    prev = [e for e in EVENTS if e <= d]
    return (d - max(prev)).days if prev else np.nan


def main(out_dir):
    out = Path(out_dir)
    snd = pd.read_parquet(out / "soundings.parquet")
    c = pd.read_parquet(out / "cases.parquet")
    c11 = c[(c["variant"] == "inv") & (c["event"] == "2011")][["rec_index", "median_q_liq_0_10", "median_Ic_0_10", "frac_sandlike_0_10"]]
    D = snd.merge(c11, on="rec_index")
    D = D[D["date"].notna() & (D["date"] < pd.Timestamp("2016-02-14"))].copy()
    D["dsl"] = D["date"].map(days_since_last)
    D = D[D["dsl"] > 0]
    D["log_dsl"] = np.log10(D["dsl"])
    D["log_q"] = np.log(D["median_q_liq_0_10"])
    D["block"] = (np.floor(D["E"] / 1000).astype(int) * 100000 + np.floor(D["N"] / 1000).astype(int))
    D["cell500"] = (np.floor(D["E"] / 500).astype(int) * 100000 + np.floor(D["N"] / 500).astype(int)).astype(str)
    D["cell250"] = (np.floor(D["E"] / 250).astype(int) * 100000 + np.floor(D["N"] / 250).astype(int)).astype(str)
    res = {"n": int(len(D)), "dsl_quantiles_days": [float(q) for q in D["dsl"].quantile([.05, .25, .5, .75, .95])]}
    for metric in ["log_q", "median_Ic_0_10", "frac_sandlike_0_10"]:
        d = D.dropna(subset=[metric])
        r = {}
        for label, f in [("pooled", f"{metric} ~ log_dsl"),
                         ("cell500_FE", f"{metric} ~ log_dsl + C(cell500)"),
                         ("cell250_FE", f"{metric} ~ log_dsl + C(cell250)")]:
            m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["block"]})
            r[label] = {"slope_per_decade": float(m.params["log_dsl"]), "se": float(m.bse["log_dsl"]),
                        "p": float(m.pvalues["log_dsl"]), "n": int(m.nobs)}
        res[metric] = r
    # pairs within 20 m with different dates
    xy = D[["E", "N"]].values
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=20.0, output_type="ndarray")
    P = pd.DataFrame({"i": pairs[:, 0], "j": pairs[:, 1]})
    for col in ["log_dsl", "log_q", "median_Ic_0_10", "date"]:
        P[col + "_i"] = D[col].values[P["i"]]
        P[col + "_j"] = D[col].values[P["j"]]
    P = P[np.abs((P["date_i"] - P["date_j"]).dt.days) > 30]
    P["d_t"] = P["log_dsl_j"] - P["log_dsl_i"]
    pr = {"n_pairs": int(len(P))}
    for metric in ["log_q", "median_Ic_0_10"]:
        Q = P.dropna(subset=[metric + "_i", metric + "_j"])
        dq = Q[metric + "_j"] - Q[metric + "_i"]
        # regress difference on difference in log time (through origin, symmetric pairs)
        x = Q["d_t"].values
        y = dq.values
        b = float(np.sum(x * y) / np.sum(x * x))
        resid = y - b * x
        se = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 1) / np.sum(x * x)))
        pr[metric] = {"slope_per_decade": b, "se": se, "n": int(len(Q))}
    res["pairs_le20m"] = pr
    # programme-wise means of log_q (raw) for context
    D["prog"] = pd.cut(D["date"], [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-02-22"), pd.Timestamp("2011-06-13"),
                                    pd.Timestamp("2011-12-23"), pd.Timestamp("2014-01-01"), pd.Timestamp("2016-02-14")],
                       labels=["P0", "P1", "P2", "P3", "P4"])
    res["programme_means"] = D.groupby("prog", observed=True).agg(n=("log_q", "size"), median_q=("median_q_liq_0_10", "median"),
                                                                    median_Ic=("median_Ic_0_10", "median")).reset_index().astype({"prog": str}).to_dict(orient="records")
    (out / "apparent_recovery.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
