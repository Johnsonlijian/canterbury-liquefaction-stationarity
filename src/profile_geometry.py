"""Profile geometry by investigation period and programme: sampling interval, start depth,
maximum depth and number of sand-like points below the water table within 15 m.

Usage: python -I profile_geometry.py <mat> <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402


def main(mat_path, out_dir):
    out = Path(out_dir)
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    rows = []
    for i, r in enumerate(recs):
        d, qc, fs = CL.clean_profile(np.atleast_1d(r.depth), np.atleast_1d(r.qc), np.atleast_1d(r.fs))
        if d.size < 10:
            continue
        valid = qc > 0
        dzs = np.diff(d)
        rows.append({"rec_index": i, "dz_median": float(np.median(dzs)), "z_start": float(d[valid][0]) if valid.any() else np.nan,
                     "z_end": float(d[-1]), "n_le15": int(np.sum(valid & (d <= 15.0)))})
    G = pd.DataFrame(rows)
    P = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "programme", "period"]]
    G = G.merge(P, on="rec_index")
    G.to_parquet(out / "profile_geometry.parquet", index=False)
    pd.set_option("display.width", 200)
    summ = G.groupby(["period", "programme"], observed=True).agg(
        n=("rec_index", "size"), dz_med=("dz_median", "median"),
        dz_le_0p011=("dz_median", lambda x: float(np.mean(x <= 0.011))),
        dz_0p02=("dz_median", lambda x: float(np.mean((x > 0.011) & (x <= 0.021)))),
        dz_gt_0p021=("dz_median", lambda x: float(np.mean(x > 0.021))),
        z_start_med=("z_start", "median"), z_end_med=("z_end", "median"),
        z_end_lt10=("z_end", lambda x: float(np.mean(x < 10.0))))
    print(summ.round(3).to_string())
    (out / "profile_geometry_summary.json").write_text(summ.round(4).reset_index().to_json(orient="records", indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
