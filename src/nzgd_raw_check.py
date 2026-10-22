"""Checks on the parsed NZGD raw profiles.

1. Zips without AGS members: list their data members (csv/xls) and Investigations.csv rows.
2. Unit check against PRJ-2937 for soundings present in both (same name): median ratio of qc and
   fs on a common depth grid.
3. Pairwise distances within the repeat families (same base name).

Usage: python -I nzgd_raw_check.py <mat> <outputs_dir> <zip_dir> [<zip_dir> ...]
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def base_name(s):
    s = re.sub(r"\s+", "-", str(s).strip().upper())
    s = s.replace("CPT-", "").replace("CPT", "")
    m = re.match(r"^-?([A-Z]{3})-?(\d{2})([A-Z]*)$", s)
    return (m.group(1) + "-" + m.group(2), m.group(3) or "") if m else (s, "")


def main(mat_path, out_dir, zip_dirs):
    out = Path(out_dir)
    meta = pd.read_parquet(out / "nzgd_raw_meta.parquet").drop_duplicates("nzgd_id")
    prof = pd.read_parquet(out / "nzgd_raw_profiles.parquet")
    # 1. zips without AGS
    for zd in zip_dirs:
        for z in sorted(Path(zd).glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                if any(n.lower().endswith(".ags") for n in names):
                    continue
                inv = [n for n in names if n.endswith("Investigations.csv")]
                row = ""
                if inv:
                    df = pd.read_csv(io.BytesIO(zf.read(inv[0])))
                    row = df[["Key", "InvestigationId", "Type", "Northings", "Eastings", "FinalDepth"]].to_dict("records")
                print("NO-AGS", z.name, [n for n in names if not n.endswith((".pdf",))][:6], row)
    # 2. unit check vs PRJ-2937
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    link = pd.read_parquet(out / "nzgd_link.parquet")
    both = meta.merge(link[["rec_index", "nzgd_id"]], on="nzgd_id", how="inner")
    for _, r in both.iterrows():
        rec = recs[int(r["rec_index"])]
        dp, qp, fp = (np.atleast_1d(np.asarray(getattr(rec, k), float)) for k in ("depth", "qc", "fs"))
        p = prof[prof["nzgd_id"] == r["nzgd_id"]].sort_values("depth")
        grid = np.arange(1.0, min(dp.max(), p["depth"].max()) - 0.1, 0.05)
        if grid.size < 20:
            continue
        qi = np.interp(grid, p["depth"], p["qc_kpa"])
        fi = np.interp(grid, p["depth"], p["fs_kpa"])
        qpi = np.interp(grid, dp, qp)
        fpi = np.interp(grid, dp, fp)
        ok = (qpi > 500) & (qi > 500)
        print(f"UNITS {r['loca_id']:<14} rec {int(r['rec_index'])}: qc ratio raw/PRJ {np.median(qi[ok] / qpi[ok]):.3f}  "
              f"fs ratio {np.median(fi[ok & (fpi > 1)] / fpi[ok & (fpi > 1)]):.3f}  fs_unit {r['fs_unit']}")
    # 3. families
    meta[["fam", "suffix"]] = meta["loca_id"].apply(lambda s: pd.Series(base_name(s)))
    fam = meta[meta["fam"].str.match(r"^[A-Z]{3}-\d{2}$")]
    for f, g in fam.groupby("fam"):
        if len(g) < 2:
            continue
        g = g.sort_values("end_date")
        xy = g[["E2193", "N2193"]].values
        dmax = float(np.max(np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))))
        print(f"FAMILY {f}: n={len(g)} max sep {dmax:.2f} m :: " + "; ".join(
            f"{s or 'orig'} {d.date()} z={z:.1f}" for s, d, z in zip(g["suffix"], g["end_date"], g["final_depth"])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
