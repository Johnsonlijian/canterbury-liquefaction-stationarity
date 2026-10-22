"""Label-precision versus ground-change checks for the Darfield (2010) timing gap.

L1  Severity contrast: AUC for 'severe or lateral spreading (codes 3-5)' versus 'none (0)',
    dropping minor/moderate (codes 1-2) whose mapping is least certain, by test period.
L2  Stratify 2012-13 soundings by the February 2011 label: ground change caused by the
    2011 event can only act where the 2011 event liquefied.
L3  Spatial smoothness of labels: agreement of the 2010 and 2011 labels between nearest
    neighbours (within 25/50/100 m) by test period, as a proxy for label resolution.

Usage: python -I timing_label_checks.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

FEB11 = pd.Timestamp("2011-02-22")
JUN11 = pd.Timestamp("2011-06-13")
DEC11 = pd.Timestamp("2011-12-23")
INDICES = ["LSN", "LPI", "LPIish"]


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def period(d):
    if pd.isna(d):
        return "nodate"
    if d < FEB11:
        return "P0"
    if d < JUN11:
        return "P1"
    if d < DEC11:
        return "P2"
    if d < pd.Timestamp("2014-01-01"):
        return "P3"
    return "P4"


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_parquet(out / "cases.parquet")
    c = c[c["variant"] == "inv"].copy()
    c["period"] = c["date"].map(period)
    w = c.pivot_table(index="rec_index", columns="event", values="label")
    w.columns = [f"lab{x}" for x in w.columns]
    c = c.merge(w, left_on="rec_index", right_index=True)
    early_xy = c[(c["event"] == "2010") & (c["period"] == "P0")].drop_duplicates("rec_index")[["E", "N"]].values
    c["dist_early"] = cKDTree(early_xy).query(c[["E", "N"]].values, k=1)[0]
    res = {}

    # L1 severity contrast
    l1 = []
    for ev in ["2010", "2011"]:
        d = c[(c["event"] == ev) & c["label"].isin([0, 3, 4, 5])].copy()
        d["ysev"] = (d["label"] >= 3).astype(int)
        for per, g in d.groupby("period"):
            for foot, s in [("all", g), ("le400m", g[(g["period"] == "P0") | (g["dist_early"] <= 400)])]:
                if len(s) < 30 or s["ysev"].nunique() < 2:
                    continue
                row = {"event": ev, "period": per, "footprint": foot, "n": int(len(s)), "n_severe": int(s["ysev"].sum())}
                for ix in INDICES:
                    row[f"AUC_{ix}"] = auc(s["ysev"], s[ix])
                l1.append(row)
    res["L1_severe_vs_none"] = l1

    # L2 P3 soundings stratified by Feb 2011 label
    d = c[(c["event"] == "2010") & c["y"].notna() & c["period"].isin(["P0", "P3"])].copy()
    l2 = []
    for per, g in d.groupby("period"):
        for lab11, s in g.groupby(g["lab2011"].between(1, 5)):
            row = {"period": per, "feb2011_manifested": bool(lab11), "n": int(len(s)), "base2010": float(s["y"].mean())}
            for ix in INDICES:
                row[f"AUC2010_{ix}"] = auc(s["y"], s[ix])
            l2.append(row)
    res["L2_2010_auc_by_feb2011_label"] = l2

    # L3 neighbour label agreement by period
    l3 = []
    snd = c[c["event"] == "2010"].drop_duplicates("rec_index")[["rec_index", "E", "N", "period", "lab2010", "lab2011"]]
    tree = cKDTree(snd[["E", "N"]].values)
    dist, idx = tree.query(snd[["E", "N"]].values, k=2)
    snd["nn_dist"] = dist[:, 1]
    snd["nn_lab2010"] = snd["lab2010"].values[idx[:, 1]]
    snd["nn_lab2011"] = snd["lab2011"].values[idx[:, 1]]
    for r in [25, 50, 100]:
        s = snd[(snd["nn_dist"] <= r) & snd["lab2010"].isin(range(0, 6)) & snd["nn_lab2010"].isin(range(0, 6))]
        for per, g in s.groupby("period"):
            l3.append({"radius_m": r, "period": per, "n": int(len(g)),
                       "agree2010_binary": float(np.mean((g["lab2010"] > 0) == (g["nn_lab2010"] > 0))),
                       "agree2011_binary": float(np.mean((g["lab2011"].between(1, 5)) == (g["nn_lab2011"].between(1, 5)))),
                       "agree2010_exact": float(np.mean(g["lab2010"] == g["nn_lab2010"]))})
    res["L3_neighbour_label_agreement"] = l3
    (out / "timing_label_checks.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
