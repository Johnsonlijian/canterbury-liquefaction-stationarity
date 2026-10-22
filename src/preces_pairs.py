"""Pre-sequence (2000-2001) versus post-sequence soundings within 10 m (NZGD raw exports).

Pairs listed in Tonkin & Taylor (2015, Appendix E, sets A, E, F and I) whose AGS files are in the
local NZGD exports. The pre-sequence soundings reach about 3 m, so the comparison is limited to
saturated sand-like layers between the water table (PRJ-2937 2011 value at the nearest sounding)
and 3 m, and to all valid points in 1-3 m.

Usage: python -I preces_pairs.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from harmonised_metrics import block_average  # noqa: E402
from repeat_panel import process  # noqa: E402

PAIRS = [(8567, 5158), (8566, 14934), (7997, 1082), (8565, 44283)]


def main(mat_path, out_dir):
    out = Path(out_dir)
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    meta = pd.read_parquet(out / "nzgd_raw_meta.parquet").drop_duplicates("nzgd_id").set_index("nzgd_id")
    prof = pd.read_parquet(out / "nzgd_raw_profiles.parquet")
    geo = pd.read_parquet(out / "soundings_geo.parquet")
    tree = cKDTree(geo[["X2193", "Y2193"]].values)
    rows = []
    for pre, post in PAIRS:
        if pre not in meta.index or post not in meta.index:
            continue
        mp, mq = meta.loc[pre], meta.loc[post]
        sep = float(np.hypot(mp["E2193"] - mq["E2193"], mp["N2193"] - mq["N2193"]))
        _, j = tree.query([mq["E2193"], mq["N2193"]])
        rec = recs[int(geo.iloc[j]["rec_index"])]
        gwt, pga = float(rec.GWT.Yr2011), float(rec.PGA.Yr2011)
        P = {}
        for k in (pre, post):
            p = prof[prof["nzgd_id"] == k].sort_values("depth")
            p = p[np.isfinite(p["depth"]) & np.isfinite(p["qc_kpa"]) & np.isfinite(p["fs_kpa"])]
            d, qc, fs = CL.clean_profile(p["depth"].values, p["qc_kpa"].values, p["fs_kpa"].values)
            P[k] = process(d, qc, fs, gwt, pga)
        m = P[pre].merge(P[post], on="z", suffixes=("_a", "_b"))
        zmax = float(min(P[pre]["z"].max(), 3.0))
        w_sat = (m["z"] > gwt) & (m["z"] <= zmax) & m["valid_a"] & m["valid_b"] & (m["ic_a"] < 2.6) & (m["ic_b"] < 2.6)
        w_all = (m["z"] >= 1.0) & (m["z"] <= zmax) & m["valid_a"] & m["valid_b"]
        rows.append({"pre": pre, "post": post, "pre_date": str(mp["end_date"].date()), "post_date": str(mq["end_date"].date()),
                     "sep_m": sep, "gwt": gwt, "zmax": zmax,
                     "dlogq_sat_sand": float(np.median(np.log(m.loc[w_sat, "qc_b"]) - np.log(m.loc[w_sat, "qc_a"]))) if w_sat.sum() >= 10 else np.nan,
                     "n_sat": int(w_sat.sum()),
                     "dlogq_all_1_3": float(np.median(np.log(m.loc[w_all, "qc_b"]) - np.log(m.loc[w_all, "qc_a"]))) if w_all.sum() >= 10 else np.nan,
                     "n_all": int(w_all.sum())})
    R = pd.DataFrame(rows)
    R.to_csv(out / "preces_pairs.csv", index=False)
    print(R.round(3).to_string(index=False))
    (out / "preces_pairs.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
