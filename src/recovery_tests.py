"""Archive-wide tests of the post-earthquake 'recovery' trend in the Canterbury database.

Metrics per sounding (outputs/harmonised_metrics.parquet, outputs/yang_metrics.parquet):
  Y_rep   qc1Ncs at the minimum-FS point, IB08, measured profile, demand of the preceding event,
          z <= 15 m: the definition that reproduces Yang et al. (2026) (yang_reproduce.py)
  crit    the same on a 0.02 m grid, fixed 2011 demand, window max(1 m, GWT)-10 m, profiles
          reaching 10 m
  p10     10th percentile of qc1Ncs of sand-like points in the same window
  med     median qc1Ncs of sand-like points in the same window
All tests use ln(metric). Cells are 250 m squares in NZTM; standard errors are clustered by 1-km
block (cells nest within blocks). Exclusions follow Yang et al.: pre-drill unknown or > 2.5 m.

Tests
  A  group means (Yang groups G1-G4) with 1-km block bootstrap intervals
  B  group contrasts with 250-m cell fixed effects
  C  STAGE programme only (P0, P1, P2), cell fixed effects
  D  contrasts by February 2011 manifestation (none vs any), cell fixed effects
  E  elapsed time within inter-event windows: slope of ln(metric) per decade of days since the
     window's opening earthquake, cell fixed effects, by window and programme
  F  low- and high-Ic groups (critical-point Ic <= 2.05 or > 2.05), as in Yang et al. Fig. 5a

Usage: python -I recovery_tests.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EV = {"2010": pd.Timestamp("2010-09-04"), "2011a": pd.Timestamp("2011-02-22"), "2011b": pd.Timestamp("2011-06-13"),
      "2011c": pd.Timestamp("2011-12-23"), "2016": pd.Timestamp("2016-02-14")}
RNG = np.random.default_rng(20261011)


def load(out):
    S = pd.read_parquet(out / "soundings.parquet")[["rec_index", "date", "pd"]]
    G = pd.read_parquet(out / "soundings_geo.parquet")[["rec_index", "X2193", "Y2193"]]
    H = pd.read_parquet(out / "harmonised_metrics.parquet")
    Y = pd.read_parquet(out / "yang_metrics.parquet")
    P = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "programme", "period", "ground_improvement"]]
    C = pd.read_parquet(out / "cases.parquet")
    lab = C[C["variant"] == "meas"].pivot(index="rec_index", columns="event", values="label")
    lab.columns = [f"label_{c}" for c in lab.columns]
    D = S.merge(G, on="rec_index").merge(H, on="rec_index").merge(P, on="rec_index").merge(lab.reset_index(), on="rec_index")
    prev = np.where(D["date"] < EV["2011a"], "2010", np.where(D["date"] < EV["2016"], "2011", "2016"))
    Ym = Y.set_index("rec_index")
    D["Y_rep"] = [Ym.at[r, f"crit_q_IB08_D{p}_meas"] if r in Ym.index else np.nan for r, p in zip(D["rec_index"], prev)]
    D["Y_rep_ic"] = [Ym.at[r, f"crit_ic_IB08_D{p}_meas"] if r in Ym.index else np.nan for r, p in zip(D["rec_index"], prev)]
    D["crit"] = np.where(D["reach10"], D["crit_q_h02_D2011"], np.nan)
    D["p10"] = D["p10_q_h02_D2011"]
    D["med"] = D["med_q_h02_D2011"]
    D["G"] = pd.cut(D["date"], [pd.Timestamp("2010-09-04"), pd.Timestamp("2011-02-23"), pd.Timestamp("2012-01-01"),
                                pd.Timestamp("2013-01-01"), pd.Timestamp("2015-01-01")], right=False, labels=["G1", "G2", "G3", "G4"])
    D["cell"] = (np.floor(D["X2193"] / 250).astype("Int64").astype(str) + "_" + np.floor(D["Y2193"] / 250).astype("Int64").astype(str))
    D["block"] = (np.floor(D["X2193"] / 1000).astype("Int64").astype(str) + "_" + np.floor(D["Y2193"] / 1000).astype("Int64").astype(str))
    D["man2011"] = np.where(D["label_2011"].between(1, 5), 1, np.where(D["label_2011"] == 0, 0, np.nan))
    # Yang et al. exclusions, plus soundings at sites flagged in the NZGD as ground-improved
    ground_improved = D["ground_improvement"].astype("boolean").fillna(False)
    keep = D["pd"].notna() & (D["pd"] <= 2.5) & D["date"].notna() & ~ground_improved
    return D[keep].copy()


def fe_ols(df, y, xcols, fe="cell", cluster="block"):
    d = df[[y, fe, cluster] + xcols].dropna()
    d = d[d.groupby(fe)[y].transform("size") > 1]
    if len(d) < 20:
        return None
    Yd = d[y] - d.groupby(fe)[y].transform("mean")
    Xd = d[xcols] - d.groupby(fe)[xcols].transform("mean")
    X = Xd.values.astype(float)
    keepc = np.abs(X).sum(axis=0) > 1e-12
    X = X[:, keepc]
    cols = [c for c, k in zip(xcols, keepc) if k]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Yd.values
    u = Yd.values - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = d[cluster].values
    ng = 0
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
        ng += 1
    n, k = X.shape
    V = XtX_inv @ meat @ XtX_inv * ng / max(ng - 1, 1) * (n - 1) / max(n - k, 1)
    se = np.sqrt(np.diag(V))
    return {"coef": dict(zip(cols, b.tolist())), "se": dict(zip(cols, se.tolist())), "n": int(n),
            "n_fe_groups": int(d[fe].nunique()), "n_clusters": int(ng)}


def pct(b, se):
    return {"pct": 100 * (np.exp(b) - 1), "lo": 100 * (np.exp(b - 1.96 * se) - 1), "hi": 100 * (np.exp(b + 1.96 * se) - 1),
            "b": b, "se": se}


def block_boot_mean(df, col, nboot=1000):
    d = df[[col, "block"]].dropna()
    blocks = d["block"].unique()
    idx = {b: np.where(d["block"].values == b)[0] for b in blocks}
    vals = d[col].values
    est = []
    for _ in range(nboot):
        bs = RNG.choice(blocks, size=blocks.size, replace=True)
        ii = np.concatenate([idx[b] for b in bs])
        est.append(vals[ii].mean())
    return float(vals.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), int(len(vals))


def main(out_dir):
    out = Path(out_dir)
    D = load(out)
    metrics = ["Y_rep", "crit", "p10", "med"]
    for m in metrics:
        D[f"ln_{m}"] = np.log(D[m])
    res = {"n": int(len(D))}
    # A: group means
    res["A_group_means"] = {m: {g: block_boot_mean(D[D["G"] == g], m) for g in ["G1", "G2", "G3", "G4"]} for m in metrics}
    res["A_group_medians"] = {m: D.groupby("G", observed=True)[m].median().round(2).to_dict() for m in metrics}
    # B: cell FE contrasts
    for g in ["G2", "G3", "G4"]:
        D[f"is_{g}"] = (D["G"] == g).astype(float)
    gx = ["is_G2", "is_G3", "is_G4"]
    res["B_raw"], res["B_cellFE"] = {}, {}
    for m in metrics:
        d = D.dropna(subset=[f"ln_{m}", "G"])
        X = np.column_stack([np.ones(len(d))] + [d[c].values for c in gx])
        b = np.linalg.lstsq(X, d[f"ln_{m}"].values, rcond=None)[0]
        res["B_raw"][m] = {c: 100 * (np.exp(bb) - 1) for c, bb in zip(gx, b[1:])}
        r = fe_ols(d, f"ln_{m}", gx)
        res["B_cellFE"][m] = {c: pct(r["coef"][c], r["se"][c]) for c in r["coef"]} | {"n": r["n"], "cells": r["n_fe_groups"]}
    # C: STAGE only P0/P1/P2
    St = D[(D["programme"] == "STAGE") & D["period"].isin(["P0", "P1", "P2"])].copy()
    St["is_P1"] = (St["period"] == "P1").astype(float)
    St["is_P2"] = (St["period"] == "P2").astype(float)
    res["C_stage"] = {}
    for m in metrics:
        r = fe_ols(St, f"ln_{m}", ["is_P1", "is_P2"])
        res["C_stage"][m] = ({c: pct(r["coef"][c], r["se"][c]) for c in r["coef"]} | {"n": r["n"], "cells": r["n_fe_groups"]}) if r else None
        res["C_stage"][m + "_means"] = St.groupby("period", observed=True)[m].mean().round(2).to_dict()
    # D: by Feb 2011 manifestation
    res["D_by_man2011"] = {}
    for man in [0, 1]:
        dd = D[D["man2011"] == man]
        res["D_by_man2011"][f"man{man}"] = {}
        for m in metrics:
            r = fe_ols(dd, f"ln_{m}", gx)
            res["D_by_man2011"][f"man{man}"][m] = ({c: pct(r["coef"][c], r["se"][c]) for c in r["coef"]} | {"n": r["n"]}) if r else None
    # E: elapsed time within windows
    windows = {"P0_since_Darfield": ("2010", EV["2011a"], ["STAGE"]),
               "P1_since_22Feb2011": ("2011a", EV["2011b"], ["STAGE"]),
               "P2_since_13Jun2011": ("2011b", EV["2011c"], ["STAGE", "OTHER"]),
               "P3_since_23Dec2011": ("2011c", EV["2016"], ["POD", "TCR", "OTHER"])}
    res["E_within_window"] = {}
    for name, (ev, end, progs) in windows.items():
        w = D[(D["date"] > EV[ev]) & (D["date"] < end) & D["programme"].isin(progs)].copy()
        w["log10_days"] = np.log10((w["date"] - EV[ev]).dt.days.clip(lower=1))
        res["E_within_window"][name] = {"n": int(len(w)), "days_range": [int((w["date"] - EV[ev]).dt.days.min()), int((w["date"] - EV[ev]).dt.days.max())]}
        for m in metrics:
            r = fe_ols(w, f"ln_{m}", ["log10_days"])
            if r:
                b, se = r["coef"]["log10_days"], r["se"]["log10_days"]
                res["E_within_window"][name][m] = {"slope_per_decade": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                                                   "n_used": r["n"], "cells": r["n_fe_groups"]}
            if name.startswith("P3"):
                for prog in ["POD"]:
                    r2 = fe_ols(w[w["programme"] == prog], f"ln_{m}", ["log10_days"])
                    if r2:
                        b, se = r2["coef"]["log10_days"], r2["se"]["log10_days"]
                        res["E_within_window"][name][m + "_POD_only"] = {"slope_per_decade": b, "se": se, "n_used": r2["n"]}
    # F: low-/high-Ic groups (critical point Ic from the reproduction)
    D["icgrp"] = np.where(D["Y_rep_ic"] <= 2.05, "lowIc", "highIc")
    res["F_icgroups"] = {}
    for grp in ["lowIc", "highIc"]:
        dd = D[D["icgrp"] == grp]
        res["F_icgroups"][grp] = {"means": dd.groupby("G", observed=True)["Y_rep"].mean().round(1).to_dict(),
                                  "n": dd.groupby("G", observed=True).size().to_dict()}
        r = fe_ols(dd, "ln_Y_rep", gx)
        res["F_icgroups"][grp]["cellFE"] = {c: pct(r["coef"][c], r["se"][c]) for c in r["coef"]} if r else None
    res["F_share_lowIc_by_G"] = D.groupby("G", observed=True)["icgrp"].apply(lambda x: float(np.mean(x == "lowIc"))).round(3).to_dict()
    (out / "recovery_tests.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main(sys.argv[1])
