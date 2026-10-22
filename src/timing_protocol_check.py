"""Check measurement-protocol differences between investigation periods.

Compares, by CPT test period, the pre-drill field (pd), the first valid cone reading,
penetration depth, and the shallow (0-3 m, 3-6 m) liquefiable contribution to LSN;
then re-tests early-vs-late 2010 discrimination after (a) truncating every profile to a
common start depth and (b) restricting to soundings with a shallow first reading.

Usage: python -I timing_protocol_check.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as C  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")
JUN11 = pd.Timestamp("2011-06-13")
DEC11 = pd.Timestamp("2011-12-23")


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def period(d):
    if pd.isna(d):
        return "nodate"
    if d < FEB11:
        return "P0_Sep10-Feb11"
    if d < JUN11:
        return "P1_Feb-Jun11"
    if d < DEC11:
        return "P2_Jun-Dec11"
    if d < pd.Timestamp("2014-01-01"):
        return "P3_2012-13"
    return "P4_2015+"


def main(mat_path, out_dir):
    out = Path(out_dir)
    snd = pd.read_parquet(out / "soundings.parquet")
    cases = pd.read_parquet(out / "cases.parquet")
    snd["period"] = snd["date"].map(period)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]

    # descriptive protocol table
    desc = snd.groupby("period").agg(n=("rec_index", "size"), pd_median=("pd", "median"), pd_mean=("pd", "mean"),
                                     pd_frac_gt0=("pd", lambda x: float(np.mean(x > 0))),
                                     zfirst_median=("z_first_valid", "median"),
                                     zfirst_q75=("z_first_valid", lambda x: float(np.quantile(x.dropna(), .75))),
                                     zmax_median=("z_max", "median"), npts_median=("n_points", "median"))
    res = {"protocol_by_period": desc.reset_index().to_dict(orient="records")}

    # recompute 2010 indices with a common start depth (truncate shallower data); same sample as
    # the case table (soundings at ground-improved sites are excluded there)
    snd_cases = snd[snd["rec_index"].isin(set(cases["rec_index"]))]
    rows = []
    for z0 in [0.0, 1.0, 1.5, 2.0]:
        for _, s in snd_cases.iterrows():
            i = int(s["rec_index"])
            r = recs[i]
            d, qc, fs = C.clean_profile(r.depth, r.qc_inv, r.fs_inv)
            if d.size < 10:
                continue
            gamma, valid = C.profile_static(d, qc, fs)
            valid = valid & (d >= z0)
            idx, _ = C.process_event(d, qc, fs, gamma, valid, float(r.GWT.Yr2010), 7.1, float(r.PGA.Yr2010))
            lab = int(r.Manifestation.Yr2010)
            rows.append({"z0": z0, "rec_index": i, "LSN": idx["LSN"], "LPI": idx["LPI"], "LPIish": idx["LPIish"],
                         "y": 1 if 1 <= lab <= 5 else (0 if lab == 0 else np.nan)})
    T = pd.DataFrame(rows).merge(snd[["rec_index", "period", "E", "N", "z_first_valid"]], on="rec_index")
    T = T[T["y"].notna()]
    early_xy = snd.loc[snd["period"] == "P0_Sep10-Feb11", ["E", "N"]].values
    tree = cKDTree(early_xy)
    T["dist_early"] = tree.query(T[["E", "N"]].values, k=1)[0]
    trunc = []
    for z0, g in T.groupby("z0"):
        for per, gg in g.groupby("period"):
            for foot, sub in [("all", gg), ("le400m", gg[(gg["period"] == "P0_Sep10-Feb11") | (gg["dist_early"] <= 400)])]:
                if len(sub) < 30:
                    continue
                trunc.append({"z0": z0, "period": per, "footprint": foot, "n": int(len(sub)), "base": float(sub["y"].mean()),
                              "AUC_LSN": auc(sub["y"], sub["LSN"]), "AUC_LPI": auc(sub["y"], sub["LPI"]),
                              "AUC_LPIish": auc(sub["y"], sub["LPIish"])})
    res["truncation_test_2010"] = trunc

    # shallow first reading restriction (z_first_valid <= 0.6 m) with original indices
    c10 = cases[(cases["variant"] == "inv") & (cases["event"] == "2010") & cases["y"].notna()].merge(
        snd[["rec_index", "period", "z_first_valid"]], on="rec_index")
    c10["dist_early"] = tree.query(c10[["E", "N"]].values, k=1)[0]
    shallow = []
    for zmax_first in [0.3, 0.6, 1.0, 99]:
        sub = c10[c10["z_first_valid"] <= zmax_first]
        for per, gg in sub.groupby("period"):
            for foot, s2 in [("all", gg), ("le400m", gg[(gg["period"] == "P0_Sep10-Feb11") | (gg["dist_early"] <= 400)])]:
                if len(s2) < 30:
                    continue
                shallow.append({"first_reading_le_m": zmax_first, "period": per, "footprint": foot, "n": int(len(s2)),
                                "base": float(s2["y"].mean()), "AUC_LSN": auc(s2["y"], s2["LSN"]),
                                "AUC_LPI": auc(s2["y"], s2["LPI"]), "AUC_LPIish": auc(s2["y"], s2["LPIish"])})
    res["first_reading_restriction_2010"] = shallow
    (out / "timing_protocol_check.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:14000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
