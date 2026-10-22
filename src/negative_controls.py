"""Negative-control strata for the apparent post-earthquake change in cone resistance.

Liquefaction-induced ground change (densification or post-liquefaction recovery) should alter cone
resistance in loose saturated sands that liquefied, mostly at shallow depth. A change in programme,
equipment or procedure would instead shift all strata, including layers that cannot liquefy.
Per sounding (measured profiles, 2011 water table for the effective stress) we compute medians of:
  lnqc_sandW_1_10   ln qc, sand-like (Ic < 2.6) below the water table, 1-10 m   (liquefiable zone)
  lnqc_sandW_1_5 / _5_10
  lnqc_sand_12_20   ln qc, sand-like, 12-20 m                                   (deep sands)
  lnqc_clay_1_10    ln qc, clay-like (Ic > 2.95), 1-10 m                         (non-liquefiable)
  lnqc_dry          ln qc, 0.5 m to the water table, profiles with GWT >= 1.5 m  (unsaturated)
  lnRf_clay_1_10, lnRf_sandW_1_10  ln friction ratio in the same strata
  ic_sandW_1_10     median Ic of the liquefiable zone
Contrasts: Yang groups G2-G4 vs G1 and STAGE P1/P2 vs P0, with 250-m cell fixed effects and 1-km
block clustering (fe_ols from recovery_tests.py).

Usage: python -I negative_controls.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from recovery_tests import fe_ols, load  # noqa: E402


def strata(d, qc, fs, gwt):
    gamma, valid = CL.profile_static(d, qc, fs)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    rf = np.where(valid, np.clip(fs, 0.1, None) / np.maximum(qc, 1.0) * 100.0, np.nan)
    lq = np.log(np.where(valid, qc, np.nan))
    sand, clay = valid & (ic < 2.6), valid & (ic > 2.95)
    out = {}

    def med(arr, m, k, nmin=10):
        out[k] = float(np.nanmedian(arr[m])) if m.sum() >= nmin else np.nan

    med(lq, sand & (d > gwt) & (d >= 1) & (d <= 10), "lnqc_sandW_1_10")
    med(lq, sand & (d > gwt) & (d >= 1) & (d <= 5), "lnqc_sandW_1_5")
    med(lq, sand & (d > gwt) & (d > 5) & (d <= 10), "lnqc_sandW_5_10")
    med(lq, sand & (d >= 12) & (d <= 20), "lnqc_sand_12_20")
    med(lq, clay & (d >= 1) & (d <= 10), "lnqc_clay_1_10")
    if gwt >= 1.5:
        med(lq, valid & (d >= 0.5) & (d < gwt), "lnqc_dry", nmin=20)
    else:
        out["lnqc_dry"] = np.nan
    med(np.log(rf), clay & (d >= 1) & (d <= 10), "lnRf_clay_1_10")
    med(np.log(rf), sand & (d > gwt) & (d >= 1) & (d <= 10), "lnRf_sandW_1_10")
    med(ic, sand & (d > gwt) & (d >= 1) & (d <= 10), "ic_sandW_1_10")
    med(np.log(np.where(valid, np.clip(fs, 0.1, None), np.nan)), sand & (d > gwt) & (d >= 1) & (d <= 10), "lnfs_sandW_1_10")
    return out


def main(mat_path, out_dir):
    out = Path(out_dir)
    f = out / "negative_controls_strata.parquet"
    if not f.exists():
        recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
        rows = []
        for i, r in enumerate(recs):
            d, qc, fs = CL.clean_profile(np.atleast_1d(r.depth), np.atleast_1d(r.qc), np.atleast_1d(r.fs))
            if d.size < 10:
                continue
            rows.append({"rec_index": i, **strata(d, qc, fs, float(r.GWT.Yr2011))})
        pd.DataFrame(rows).to_parquet(f, index=False)
    N = pd.read_parquet(f)
    D = load(out).merge(N, on="rec_index", how="left")
    cols = [c for c in N.columns if c != "rec_index"]
    for g in ["G2", "G3", "G4"]:
        D[f"is_{g}"] = (D["G"] == g).astype(float)
    res = {"cellFE_G": {}, "stage_cellFE": {}, "raw_group_means": {}}
    St = D[(D["programme"] == "STAGE") & D["period"].isin(["P0", "P1", "P2"])].copy()
    St["is_P1"] = (St["period"] == "P1").astype(float)
    St["is_P2"] = (St["period"] == "P2").astype(float)
    for c in cols:
        r = fe_ols(D, c, ["is_G2", "is_G3", "is_G4"])
        res["cellFE_G"][c] = ({k: [r["coef"][k], r["coef"][k] - 1.96 * r["se"][k], r["coef"][k] + 1.96 * r["se"][k]] for k in r["coef"]}
                              | {"n": r["n"]}) if r else None
        r2 = fe_ols(St, c, ["is_P1", "is_P2"])
        res["stage_cellFE"][c] = ({k: [r2["coef"][k], r2["coef"][k] - 1.96 * r2["se"][k], r2["coef"][k] + 1.96 * r2["se"][k]] for k in r2["coef"]}
                                  | {"n": r2["n"]}) if r2 else None
        res["raw_group_means"][c] = D.groupby("G", observed=True)[c].mean().round(3).to_dict()
    # same contrasts by February 2011 manifestation for the key strata
    res["cellFE_G_by_man2011"] = {}
    for man in [0, 1]:
        dd = D[D["man2011"] == man]
        res["cellFE_G_by_man2011"][f"man{man}"] = {}
        for c in ["lnqc_sandW_1_10", "lnqc_clay_1_10", "lnqc_sand_12_20", "lnRf_clay_1_10", "lnqc_dry"]:
            r = fe_ols(dd, c, ["is_G2", "is_G3", "is_G4"])
            res["cellFE_G_by_man2011"][f"man{man}"][c] = ({k: [r["coef"][k], r["coef"][k] - 1.96 * r["se"][k], r["coef"][k] + 1.96 * r["se"][k]]
                                                           for k in r["coef"]} | {"n": r["n"]}) if r else None
    (out / "negative_controls.json").write_text(json.dumps(res, indent=1, default=float))
    for c in cols:
        a = res["cellFE_G"][c]
        s = res["stage_cellFE"][c]
        if a:
            print(f"{c:<18} G4-G1 {a['is_G4'][0]:+.3f} [{a['is_G4'][1]:+.3f},{a['is_G4'][2]:+.3f}]  G3-G1 {a['is_G3'][0]:+.3f}  "
                  f"G2-G1 {a['is_G2'][0]:+.3f} [{a['is_G2'][1]:+.3f},{a['is_G2'][2]:+.3f}] n={a['n']}  | STAGE P2-P0 "
                  + (f"{s['is_P2'][0]:+.3f} [{s['is_P2'][1]:+.3f},{s['is_P2'][2]:+.3f}] n={s['n']}" if s else "NA"))
    print(json.dumps(res["cellFE_G_by_man2011"], indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
