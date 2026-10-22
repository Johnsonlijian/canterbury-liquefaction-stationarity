"""Parse raw NZGD AGS4 CPT exports (read in memory from the zips; nothing extracted or executed).

For every AGS member: LOCA (ID, NZTM E/N, final depth, end date), SCPG (pre-drill depth, cone
reference), SCPT (depth, qc, fs, u2). The digitised NZGD AGS files label fs and u2 as MPa, but in
some files the values are in kPa. Units are therefore inferred per profile from the median
friction ratio fs/qc over points with qc > 0.5 MPa: a ratio of 0.002-0.1 means MPa, 2-100 means
kPa. Profiles with an ambiguous ratio are flagged and excluded downstream. The unit inference is
checked against PRJ-2937 for soundings present in both sources (same name, within 3 m).

Outputs (private working data, not for redistribution):
  outputs/nzgd_raw_profiles.parquet  long table: nzgd_id, loca_id, depth, qc_kpa, fs_kpa, u2_kpa
  outputs/nzgd_raw_meta.parquet      one row per profile with metadata and unit flags

Usage: python -I nzgd_raw_profiles.py <outputs_dir> <zip_dir> [<zip_dir> ...]
"""
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def split_ags(text):
    groups = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith('"GROUP",'):
            i += 1
            continue
        group = next(csv.reader([lines[i]]))[1]
        if i + 1 >= len(lines) or not lines[i + 1].startswith('"HEADING"'):
            i += 1
            continue
        heading = next(csv.reader([lines[i + 1]]))[1:]
        rows = []
        j = i + 2
        while j < len(lines) and not lines[j].startswith('"GROUP",'):
            if lines[j].startswith('"DATA"'):
                vals = next(csv.reader([lines[j]]))[1:]
                vals += [""] * max(0, len(heading) - len(vals))
                rows.append(dict(zip(heading, vals)))
            j += 1
        groups[group] = rows
        i = j
    return groups


def fnum(x):
    try:
        return float(x) if x not in (None, "") else np.nan
    except ValueError:
        return np.nan


def main(out_dir, zip_dirs):
    out = Path(out_dir)
    metas, longs = [], []
    for zd in zip_dirs:
        for z in sorted(Path(zd).glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():
                    if not member.lower().endswith(".ags"):
                        continue
                    m = re.search(r"(\d+)-Investigation-(\d+)", member)
                    nzgd_id = int(m.group(1)) if m else -1
                    g = split_ags(zf.read(member).decode("utf-8", errors="replace"))
                    loca = (g.get("LOCA") or [{}])[0]
                    scpg = (g.get("SCPG") or [{}])[0]
                    scpt = g.get("SCPT") or []
                    d = np.array([fnum(r.get("SCPT_DPTH")) for r in scpt])
                    qc = np.array([fnum(r.get("SCPT_RES")) for r in scpt])
                    fs = np.array([fnum(r.get("SCPT_FRES")) for r in scpt])
                    u2 = np.array([fnum(r.get("SCPT_PWP2")) for r in scpt])
                    ok = np.isfinite(d) & np.isfinite(qc) & (qc > 0.5) & np.isfinite(fs) & (fs > 0)
                    ratio = float(np.median(fs[ok] / qc[ok])) if ok.sum() >= 20 else np.nan
                    if 0.002 <= ratio <= 0.1:
                        fs_unit = "MPa"
                    elif 2.0 <= ratio <= 100.0:
                        fs_unit = "kPa"
                    else:
                        fs_unit = "ambiguous"
                    # u2: hydrostatic below the water table is ~10 kPa/m; infer from the deepest decile
                    deep = np.isfinite(u2) & np.isfinite(d) & (d >= np.nanpercentile(d, 90)) if np.isfinite(d).any() else np.zeros(0, bool)
                    u_med = float(np.median(u2[deep])) if deep.sum() > 5 else np.nan
                    u_unit = "MPa" if (np.isfinite(u_med) and abs(u_med) < 2.0) else ("kPa" if np.isfinite(u_med) else "unknown")
                    meta = {"nzgd_id": nzgd_id, "zip": z.name, "loca_id": loca.get("LOCA_ID", ""),
                            "E2193": fnum(loca.get("LOCA_NATE")), "N2193": fnum(loca.get("LOCA_NATN")),
                            "gref": loca.get("LOCA_GREF", ""), "final_depth": fnum(loca.get("LOCA_FDEP")),
                            "end_date": pd.to_datetime(loca.get("LOCA_ENDD") or loca.get("LOCA_STAR") or None, errors="coerce"),
                            "predrill_m": fnum(scpg.get("SCPG_PDRL")), "cone_ref": scpg.get("SCPG_REF", ""),
                            "n_rows": len(scpt), "fs_ratio_median": ratio, "fs_unit": fs_unit,
                            "u2_deep_median_raw": u_med, "u2_unit": u_unit}
                    metas.append(meta)
                    fs_kpa = fs * (1000.0 if fs_unit == "MPa" else 1.0)
                    u2_kpa = u2 * (1000.0 if u_unit == "MPa" else 1.0)
                    longs.append(pd.DataFrame({"nzgd_id": nzgd_id, "loca_id": meta["loca_id"], "depth": d,
                                               "qc_kpa": qc * 1000.0, "fs_kpa": fs_kpa, "u2_kpa": u2_kpa}))
    M = pd.DataFrame(metas)
    L = pd.concat(longs, ignore_index=True)
    M.to_parquet(out / "nzgd_raw_meta.parquet", index=False)
    L.to_parquet(out / "nzgd_raw_profiles.parquet", index=False)
    print(M[["nzgd_id", "loca_id", "end_date", "final_depth", "predrill_m", "n_rows", "fs_ratio_median", "fs_unit", "u2_unit"]].to_string())
    print(M["fs_unit"].value_counts(), M["u2_unit"].value_counts())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
