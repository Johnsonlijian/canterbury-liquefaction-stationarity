"""Robustness of the within-window elapsed-time slope of shallow sand resistance to the scale of
location control.

For the P2 (13 Jun-22 Dec 2011) and P3 (2012-2013) windows: slope of ln qc of saturated sand-like
layers (1-5 m, 5-10 m) per decade of days since the opening earthquake, with
  (a) cell fixed effects at 500, 250, 100 and 50 m;
  (b) first differences of pairs within 20 m and 50 m tested in the same window, optionally
      requiring the same programme, no ground improvement and a pre-drill difference <= 0.5 m.
Pairs use each sounding at most in its nearest-neighbour pair to limit dependence.

Usage: python -I window_robustness.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_tests import EV, fe_ols, load  # noqa: E402


def cells(D, size):
    return np.floor(D["X2193"] / size).astype(int).astype(str) + "_" + np.floor(D["Y2193"] / size).astype(int).astype(str)


def pair_fd(w, col, rmax, same_prog, pd_tol, ev):
    w = w.dropna(subset=[col]).reset_index(drop=True)
    if len(w) < 10:
        return None
    tree = cKDTree(w[["X2193", "Y2193"]].values)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    if pairs.size == 0:
        return None
    a, b = pairs[:, 0], pairs[:, 1]
    keep = np.ones(len(a), bool)
    if same_prog:
        keep &= w.loc[a, "programme"].values == w.loc[b, "programme"].values
    keep &= ~(w.loc[a, "ground_improvement"].fillna(False).values.astype(bool) | w.loc[b, "ground_improvement"].fillna(False).values.astype(bool))
    keep &= np.abs(w.loc[a, "pd"].values - w.loc[b, "pd"].values) <= pd_tol
    da = (w.loc[a, "date"].values - np.datetime64(EV[ev])) / np.timedelta64(1, "D")
    db = (w.loc[b, "date"].values - np.datetime64(EV[ev])) / np.timedelta64(1, "D")
    keep &= np.abs(da - db) >= 30
    a, b, da, db = a[keep], b[keep], da[keep], db[keep]
    if len(a) < 10:
        return {"n_pairs": int(len(a))}
    # orient: later minus earlier
    later_b = db > da
    dy = np.where(later_b, w.loc[b, col].values - w.loc[a, col].values, w.loc[a, col].values - w.loc[b, col].values)
    dx = np.abs(np.log10(np.maximum(db, 1)) - np.log10(np.maximum(da, 1)))
    # each sounding used once: greedy by distance
    dist = np.hypot(w.loc[a, "X2193"].values - w.loc[b, "X2193"].values, w.loc[a, "Y2193"].values - w.loc[b, "Y2193"].values)
    used, sel = set(), []
    for i in np.argsort(dist):
        if a[i] in used or b[i] in used:
            continue
        used.update([a[i], b[i]])
        sel.append(i)
    sel = np.array(sel)
    x, y = dx[sel], dy[sel]
    slope = float(np.sum(x * y) / np.sum(x * x))  # through the origin: no change without elapsed time
    resid = y - slope * x
    se = float(np.sqrt(np.sum((x * resid) ** 2)) / np.sum(x * x))
    return {"n_pairs": int(len(sel)), "slope": slope, "lo": slope - 1.96 * se, "hi": slope + 1.96 * se,
            "median_dy": float(np.median(y)), "median_dx": float(np.median(x))}


def main(out_dir):
    out = Path(out_dir)
    D = load(out).merge(pd.read_parquet(out / "negative_controls_strata.parquet"), on="rec_index", how="left")
    windows = {"P2": ("2011b", EV["2011c"], ["STAGE", "OTHER"]), "P3": ("2011c", EV["2016"], ["POD", "TCR", "OTHER"])}
    res = {}
    for name, (ev, end, progs) in windows.items():
        w = D[(D["date"] > EV[ev]) & (D["date"] < end) & D["programme"].isin(progs)].copy()
        w["log10_days"] = np.log10((w["date"] - EV[ev]).dt.days.clip(lower=1))
        res[name] = {}
        for col in ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sand_12_20"]:
            r_ = {}
            for size in [500, 250, 100, 50]:
                w["cellx"] = cells(w, size)
                r = fe_ols(w, col, ["log10_days"], fe="cellx")
                if r:
                    b, se = r["coef"]["log10_days"], r["se"]["log10_days"]
                    r_[f"cellFE_{size}m"] = {"slope": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "n": r["n"], "cells": r["n_fe_groups"]}
            for rmax in [20, 50]:
                for same_prog, pd_tol, tag in [(False, 99.0, "all"), (True, 0.5, "sameprog_pd")]:
                    r_[f"pairs_{rmax}m_{tag}"] = pair_fd(w, col, rmax, same_prog, pd_tol, ev)
            res[name][col] = r_
    (out / "window_robustness.json").write_text(json.dumps(res, indent=1, default=float))
    for name in res:
        for col in res[name]:
            print(name, col)
            for k, v in res[name][col].items():
                if v and "slope" in v:
                    print(f"   {k:<26} {v['slope']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]  n={v.get('n', v.get('n_pairs'))}")
                else:
                    print(f"   {k:<26} {v}")


if __name__ == "__main__":
    main(sys.argv[1])
