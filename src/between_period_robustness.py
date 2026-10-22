"""Robustness of the between-period change in cone resistance to the scale of location control.

Contrasts of each stratum (negative_controls_strata.parquet) between soundings pushed before the
22 February 2011 earthquake (P0) and soundings pushed in 2012-2014 (G3-G4), and between P0 and
P1/P2 (2011, STAGE programme):
  (a) cell fixed effects at 500, 250, 100, 50 and 25 m;
  (b) nearest-neighbour pairs (each sounding used once) within 10, 20 and 50 m, later minus
      earlier, with 1-km block bootstrap intervals; optionally requiring pre-drill difference
      <= 0.5 m and excluding ground-improvement sites.
Also reported for the pairs: separation and the 2011 manifestation label of the later sounding.

Usage: python -I between_period_robustness.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_tests import fe_ols, load  # noqa: E402

RNG = np.random.default_rng(7)
STRATA = ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sandW_1_10", "lnqc_sand_12_20", "lnqc_clay_1_10", "lnqc_dry",
          "lnRf_sandW_1_10", "ic_sandW_1_10"]


def cells(D, size):
    return np.floor(D["X2193"] / size).astype(int).astype(str) + "_" + np.floor(D["Y2193"] / size).astype(int).astype(str)


def nn_pairs(early, late, rmax):
    tree = cKDTree(early[["X2193", "Y2193"]].values)
    d, j = tree.query(late[["X2193", "Y2193"]].values, distance_upper_bound=rmax)
    cand = [(dd, i, jj) for i, (dd, jj) in enumerate(zip(d, j)) if np.isfinite(dd)]
    used_e, used_l, out = set(), set(), []
    for dd, i, jj in sorted(cand):
        if i in used_l or jj in used_e:
            continue
        used_l.add(i)
        used_e.add(jj)
        out.append((i, jj, dd))
    return out


def boot(y, blocks, n=2000):
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    est = []
    for _ in range(n):
        s = RNG.choice(ub, ub.size, replace=True)
        ii = np.concatenate([idx[b] for b in s])
        est.append(np.median(y[ii]))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main(out_dir):
    out = Path(out_dir)
    D = load(out).merge(pd.read_parquet(out / "negative_controls_strata.parquet"), on="rec_index", how="left")
    D["grp"] = np.where(D["period"] == "P0", "P0", np.where(D["period"].isin(["P1", "P2"]), "P12",
                        np.where(D["G"].isin(["G3", "G4"]), "G34", "other")))
    res = {"cellFE": {}, "pairs": {}}
    for contrast, (g0, g1) in {"G34_vs_P0": ("P0", "G34"), "P12_vs_P0": ("P0", "P12")}.items():
        S = D[D["grp"].isin([g0, g1])].copy()
        S["is_late"] = (S["grp"] == g1).astype(float)
        res["cellFE"][contrast] = {}
        for c in STRATA:
            rr = {}
            for size in [500, 250, 100, 50, 25]:
                S["cellx"] = cells(S, size)
                r = fe_ols(S, c, ["is_late"], fe="cellx")
                if r and "is_late" in r["coef"]:
                    b, se = r["coef"]["is_late"], r["se"]["is_late"]
                    rr[f"{size}m"] = [b, b - 1.96 * se, b + 1.96 * se, r["n"], r["n_fe_groups"]]
            res["cellFE"][contrast][c] = rr
        early, late = S[S["grp"] == g0].reset_index(drop=True), S[S["grp"] == g1].reset_index(drop=True)
        res["pairs"][contrast] = {}
        for rmax in [10, 20, 50]:
            pr = nn_pairs(early, late, rmax)
            if not pr:
                continue
            li = np.array([p[0] for p in pr])
            ei = np.array([p[1] for p in pr])
            sep = np.array([p[2] for p in pr])
            ok_pd = np.abs(late.loc[li, "pd"].values - early.loc[ei, "pd"].values) <= 0.5
            ok_gi = ~(late.loc[li, "ground_improvement"].fillna(False).values.astype(bool))
            blocks = late.loc[li, "block"].values
            rp = {"n_pairs": int(len(pr)), "median_sep_m": float(np.median(sep)),
                  "share_late_man2011": float(np.nanmean(late.loc[li, "man2011"].values))}
            for c in STRATA:
                y = late.loc[li, c].values - early.loc[ei, c].values
                for tag, m in [("all", np.isfinite(y)), ("pd_gi_ok", np.isfinite(y) & ok_pd & ok_gi)]:
                    if m.sum() >= 8:
                        lo, hi = boot(y[m], blocks[m])
                        rp[f"{c}_{tag}"] = [float(np.median(y[m])), lo, hi, int(m.sum())]
            res["pairs"][contrast][f"r{rmax}"] = rp
    (out / "between_period_robustness.json").write_text(json.dumps(res, indent=1, default=float))
    for contrast in res["cellFE"]:
        print("==", contrast, "cell FE")
        for c, rr in res["cellFE"][contrast].items():
            print(f"  {c:<17} " + "  ".join(f"{k}:{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]n{v[3]}" for k, v in rr.items()))
        print("==", contrast, "pairs (median later-earlier)")
        for rk, rp in res["pairs"][contrast].items():
            print(f"  {rk}: n={rp['n_pairs']} sep={rp['median_sep_m']:.1f} man2011={rp['share_late_man2011']:.2f}")
            for c in STRATA:
                for tag in ["all", "pd_gi_ok"]:
                    v = rp.get(f"{c}_{tag}")
                    if v:
                        print(f"     {c:<17} {tag:<9} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] n={v[3]}")


if __name__ == "__main__":
    main(sys.argv[1])
