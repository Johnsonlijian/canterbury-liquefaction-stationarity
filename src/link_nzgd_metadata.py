"""Link PRJ-2937 soundings to NZGD catalogue records (organisation, project, ground improvement).

Matching rule: nearest NZGD CPT record within 3 m (NZTM) whose End Date is within 2 days of
the decoded PRJ-2937 test date; ties broken by final-depth difference. Unmatched soundings are
kept with empty metadata. The NZGD export is used only as a private working index.

Usage: python -I link_nzgd_metadata.py <outputs_dir> <nzgd_urban_all.csv>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def main(out_dir, nzgd_csv):
    out = Path(out_dir)
    snd = pd.read_parquet(out / "soundings.parquet").merge(pd.read_parquet(out / "soundings_geo.parquet"), on="rec_index")
    nz = pd.read_csv(nzgd_csv)
    nz = nz[nz["Investigation Type"].astype(str).str.startswith("CPT")].copy()
    nz["end_date"] = pd.to_datetime(nz["End Date"], format="%d/%m/%Y", errors="coerce")
    tree = cKDTree(nz[["NZTM Easting", "NZTM Northing"]].values)
    rows = []
    for _, s in snd.iterrows():
        idx = tree.query_ball_point([s["X2193"], s["Y2193"]], r=3.0)
        best = None
        for j in idx:
            r = nz.iloc[j]
            dd = abs((r["end_date"] - s["date"]).days) if pd.notna(r["end_date"]) and pd.notna(s["date"]) else 999
            dist = float(np.hypot(r["NZTM Easting"] - s["X2193"], r["NZTM Northing"] - s["Y2193"]))
            ddep = abs(float(r["Final Depth"]) - float(s["z_max"])) if pd.notna(r["Final Depth"]) else 99
            key = (dd > 2, ddep, dist)
            if best is None or key < best[0]:
                best = (key, r, dist, dd, ddep)
        row = {"rec_index": int(s["rec_index"])}
        if best is not None and best[3] <= 2:
            _, r, dist, dd, ddep = best
            row.update({"nzgd_id": int(r["NZGD ID"]), "nzgd_name": r["Investigation ID"], "organisation": r["Organisation"],
                        "project": r["Project Name"], "ground_improvement": bool(r["Has Ground Improvement Been Conducted?"]),
                        "ground_level": r["Ground Level"], "vertical_datum": r["Vertical Datum"],
                        "match_dist_m": dist, "match_date_diff_d": dd, "match_depth_diff_m": ddep})
        rows.append(row)
    L = pd.DataFrame(rows)
    L.to_parquet(out / "nzgd_link.parquet", index=False)
    m = L["nzgd_id"].notna()
    res = {"n_soundings": int(len(L)), "matched": int(m.sum()), "match_rate": float(m.mean()),
           "median_dist_m": float(L.loc[m, "match_dist_m"].median()),
           "depth_diff_le_0.5m": float((L.loc[m, "match_depth_diff_m"] <= 0.5).mean()),
           "organisations_top": L["organisation"].value_counts().head(15).to_dict(),
           "ground_improvement_true": int(L["ground_improvement"].fillna(False).sum())}
    S = snd.merge(L, on="rec_index", how="left")
    S["period"] = pd.cut(S["date"], [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-02-22"), pd.Timestamp("2011-06-13"),
                                    pd.Timestamp("2011-12-23"), pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")],
                         right=False, labels=["P0", "P1", "P2", "P3", "P4"])
    res["organisation_by_period"] = {str(p): g["organisation"].value_counts().head(6).to_dict() for p, g in S.groupby("period", observed=True)}
    res["project_prefix_by_period"] = {str(p): g["project"].astype(str).str.split(",").str[0].value_counts().head(6).to_dict()
                                       for p, g in S.groupby("period", observed=True)}
    (out / "nzgd_link_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
