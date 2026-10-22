"""Decompose the apparent post-earthquake rise in CPT resistance.

For every sounding, depth-band statistics are computed from the raw (measured) profile:
  bands 1-4 m, 4-8 m, 8-12 m, 12-20 m; within each band, for sand-like points (Ic < 2.6)
  the median raw qc (MPa) and median qc1Ncs (2011 water table), plus median Ic (all points).
Pairs of soundings <= R m apart (R = 10, 20) tested > 30 days apart are formed. For each
pair (ordered early -> late):
  * intervening = number of major events (22 Feb 2011, 13 Jun 2011, 23 Dec 2011) between
    the two test dates; liq_between = site manifested (PRJ-2937 label) in the February 2011
    event when it lies between the two tests.
  * outcome = late - early log(median qc) per band.
Regressions (cluster-robust by 1 km block):
  d_logq ~ d_logdays_since_event          [no intervening event: pure elapsed-time effect]
  d_logq ~ intervening + liq_between       [pairs spanning the February 2011 event]
Deep bands act as an internal control for programme/equipment differences: ground change
from liquefaction should be concentrated in shallow liquefiable layers.

Usage: python -I recovery_decomposition.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402

EVENTS = [pd.Timestamp(x) for x in ["2010-09-04", "2011-02-22", "2011-06-13", "2011-12-23"]]
BANDS = [(1, 4), (4, 8), (8, 12), (12, 20)]


def band_stats(r):
    d, qc, fs = CL.clean_profile(r.depth, r.qc, r.fs)
    if d.size < 10:
        return None
    gamma, valid = CL.profile_static(d, qc, fs)
    gwt = float(r.GWT.Yr2011)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    q = CL.qc1ncs_bi2014(np.where(valid, qc, 1.0), sev, CL.fines_content(ic))
    out = {}
    for lo, hi in BANDS:
        m = valid & (d >= lo) & (d < hi)
        ms = m & (ic < CL.IC_LIQ)
        key = f"{lo}_{hi}"
        out[f"qc_sand_{key}"] = float(np.median(qc[ms]) / 1000.0) if ms.sum() >= 10 else np.nan
        out[f"q1n_sand_{key}"] = float(np.median(q[ms])) if ms.sum() >= 10 else np.nan
        out[f"ic_{key}"] = float(np.median(ic[m])) if m.sum() >= 10 else np.nan
        out[f"fsand_{key}"] = float(np.mean(ic[m] < CL.IC_LIQ)) if m.sum() >= 10 else np.nan
    return out


def days_since_last(d):
    prev = [e for e in EVENTS if e <= d]
    return (d - max(prev)).days if prev else np.nan


def n_between(a, b):
    lo, hi = min(a, b), max(a, b)
    return int(sum(1 for e in EVENTS[1:] if lo < e <= hi))


def main(mat_path, out_dir):
    out = Path(out_dir)
    snd = pd.read_parquet(out / "soundings.parquet")
    lab = pd.read_parquet(out / "wide_inv.parquet")[["sev2010", "sev2011", "sev2016"]]
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    rows = []
    for _, s in snd.iterrows():
        bs = band_stats(recs[int(s["rec_index"])])
        if bs is None:
            continue
        bs["rec_index"] = int(s["rec_index"])
        rows.append(bs)
    B = snd[["rec_index", "E", "N", "date"]].merge(pd.DataFrame(rows), on="rec_index").merge(lab, left_on="rec_index", right_index=True, how="left")
    B = B[B["date"].notna() & (B["date"] < pd.Timestamp("2016-02-14"))].reset_index(drop=True)
    B["dsl"] = B["date"].map(days_since_last)
    B.to_parquet(out / "depth_band_stats.parquet", index=False)
    res = {"n_soundings": int(len(B))}
    tree = cKDTree(B[["E", "N"]].values)
    for R in [10, 20]:
        pr = tree.query_pairs(r=float(R), output_type="ndarray")
        P = pd.DataFrame({"a": pr[:, 0], "b": pr[:, 1]})
        da, db = B["date"].values[P["a"]], B["date"].values[P["b"]]
        swap = db < da
        P.loc[swap, ["a", "b"]] = P.loc[swap, ["b", "a"]].values
        P["early"], P["late"] = P["a"], P["b"]
        P["dt_days"] = (B["date"].values[P["late"]] - B["date"].values[P["early"]]).astype("timedelta64[D]").astype(int)
        P = P[P["dt_days"] > 30].copy()
        P["n_between"] = [n_between(B["date"].iloc[i], B["date"].iloc[j]) for i, j in zip(P["early"], P["late"])]
        P["d_logdsl"] = np.log10(B["dsl"].values[P["late"]]) - np.log10(B["dsl"].values[P["early"]])
        P["dist"] = np.hypot(B["E"].values[P["early"]] - B["E"].values[P["late"]], B["N"].values[P["early"]] - B["N"].values[P["late"]])
        P["sev2011"] = B["sev2011"].values[P["early"]]
        P["spans_feb2011"] = [(B["date"].iloc[i] < EVENTS[1] <= B["date"].iloc[j]) for i, j in zip(P["early"], P["late"])]
        P["liq2011"] = P["sev2011"].between(1, 5).astype(int)
        P["block"] = (np.floor(B["E"].values[P["early"]] / 1000).astype(int) * 100000 + np.floor(B["N"].values[P["early"]] / 1000).astype(int))
        rr = {"n_pairs": int(len(P)), "n_no_intervening": int((P["n_between"] == 0).sum()),
              "n_span_feb2011": int(P["spans_feb2011"].sum())}
        for lo, hi in BANDS:
            key = f"{lo}_{hi}"
            for var in ["qc_sand", "q1n_sand", "ic"]:
                col = f"{var}_{key}"
                y = (np.log(B[col].values[P["late"]]) - np.log(B[col].values[P["early"]])) if var != "ic" else (B[col].values[P["late"]] - B[col].values[P["early"]])
                P["y"] = y
                Q = P.dropna(subset=["y"])
                Q = Q[np.isfinite(Q["y"])]
                ent = {}
                q0 = Q[Q["n_between"] == 0]
                if len(q0) > 30 and q0["d_logdsl"].abs().sum() > 0:
                    m0 = smf.ols("y ~ d_logdsl", data=q0).fit(cov_type="cluster", cov_kwds={"groups": q0["block"]})
                    ent["no_intervening_slope_per_decade"] = [float(m0.params["d_logdsl"]), float(m0.bse["d_logdsl"]), int(m0.nobs)]
                    ent["no_intervening_intercept"] = [float(m0.params["Intercept"]), float(m0.bse["Intercept"])]
                q1 = Q[Q["spans_feb2011"]]
                if len(q1) > 30 and q1["liq2011"].nunique() > 1:
                    m1 = smf.ols("y ~ liq2011", data=q1).fit(cov_type="cluster", cov_kwds={"groups": q1["block"]})
                    ent["span_feb2011_mean_change_nonliq"] = [float(m1.params["Intercept"]), float(m1.bse["Intercept"])]
                    ent["span_feb2011_extra_change_if_liq"] = [float(m1.params["liq2011"]), float(m1.bse["liq2011"]), float(m1.pvalues["liq2011"]), int(m1.nobs)]
                q3 = Q[Q["n_between"] >= 1]
                if len(q3) > 30:
                    m3 = smf.ols("y ~ C(n_between)", data=q3).fit(cov_type="cluster", cov_kwds={"groups": q3["block"]})
                    ent["by_n_between_params"] = {k: [float(v), float(m3.bse[k])] for k, v in m3.params.items()}
                rr[f"{var}_{key}"] = ent
        res[f"R{R}"] = rr
    (out / "recovery_decomposition.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:16000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
