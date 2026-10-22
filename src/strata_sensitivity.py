"""Sensitivity of the between-period strata contrasts (Paper B) to the profile variant and to the
water table used to classify saturated layers.

Variants: measured profiles with the 2011 water table (primary), inverse-filtered profiles with
the 2011 water table, and measured profiles with the 2016 water table. For each, the contrast
between soundings pushed in 2012-2014 and soundings pushed before 22 February 2011 is estimated
with 100-m and 50-m cell fixed effects and in nearest-neighbour pairs within 50 m.

Usage: python -I strata_sensitivity.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from between_period_robustness import boot, cells, nn_pairs  # noqa: E402
from negative_controls import strata  # noqa: E402
from recovery_tests import fe_ols, load  # noqa: E402

KEYS = ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sand_12_20", "lnqc_dry", "lnRf_sandW_1_10"]


def compute(mat_path, out):
    f = out / "strata_variants.parquet"
    if f.exists():
        return pd.read_parquet(f)
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    rows = []
    for i, r in enumerate(recs):
        for tag, (qk, fk, gk) in {"inv_gwt2011": ("qc_inv", "fs_inv", "Yr2011"), "meas_gwt2016": ("qc", "fs", "Yr2016")}.items():
            d, qc, fs = CL.clean_profile(np.atleast_1d(r.depth), np.atleast_1d(getattr(r, qk)), np.atleast_1d(getattr(r, fk)))
            if d.size < 10:
                continue
            s = strata(d, qc, fs, float(getattr(r.GWT, gk)))
            rows.append({"rec_index": i, "variant": tag, **{k: s[k] for k in KEYS}})
    V = pd.DataFrame(rows)
    V.to_parquet(f, index=False)
    return V


def contrasts(D):
    D = D.copy()
    D["grp"] = np.where(D["period"] == "P0", "P0", np.where(D["G"].isin(["G3", "G4"]), "G34", "other"))
    S = D[D["grp"].isin(["P0", "G34"])].copy()
    S["is_late"] = (S["grp"] == "G34").astype(float)
    res = {}
    early, late = S[S["grp"] == "P0"].reset_index(drop=True), S[S["grp"] == "G34"].reset_index(drop=True)
    pr = nn_pairs(early, late, 50)
    li, ei = np.array([p[0] for p in pr]), np.array([p[1] for p in pr])
    for k in KEYS:
        res[k] = {}
        for size in [100, 50]:
            S["cellx"] = cells(S, size)
            r = fe_ols(S, k, ["is_late"], fe="cellx")
            b, se = r["coef"]["is_late"], r["se"]["is_late"]
            res[k][f"{size}m"] = [b, b - 1.96 * se, b + 1.96 * se, r["n"]]
        y = late.loc[li, k].values - early.loc[ei, k].values
        m = np.isfinite(y)
        lo, hi = boot(y[m], late.loc[li, "block"].values[m])
        res[k]["pairs50"] = [float(np.median(y[m])), lo, hi, int(m.sum())]
    return res


def main(mat_path, out_dir):
    out = Path(out_dir)
    base = load(out)
    res = {"meas_gwt2011": contrasts(base.merge(pd.read_parquet(out / "negative_controls_strata.parquet")[["rec_index"] + KEYS], on="rec_index"))}
    V = compute(mat_path, out)
    for tag in ["inv_gwt2011", "meas_gwt2016"]:
        res[tag] = contrasts(base.merge(V[V["variant"] == tag].drop(columns="variant"), on="rec_index"))
    (out / "strata_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))
    for tag, r in res.items():
        print("==", tag)
        for k, v in r.items():
            print(f"   {k:<17} " + "  ".join(f"{s}: {x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}] n={x[3]}" for s, x in v.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
