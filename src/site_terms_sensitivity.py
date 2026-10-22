"""Sensitivity of the site-term and state-dependence results to the CPT profile variant
(measured vs inverse-filtered) and to excluding lateral-spreading labels (codes 4-5).

Usage: python -I site_terms_sensitivity.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_terms_glmm import fit  # noqa: E402
import state_dependence as SD  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")


def main(out_dir):
    out = Path(out_dir)
    allc = pd.read_parquet(out / "cases.parquet")
    res = {}
    for variant in ["meas", "inv"]:
        for drop_ls in [False, True]:
            c = allc[(allc["variant"] == variant) & allc["y"].notna()].copy()
            if drop_ls:
                c = c[~c["label"].isin([4, 5])]
            c["p0"] = (c["date"] < FEB11).astype(int)
            key = f"{variant}{'_noLS' if drop_ls else ''}"
            if variant == "inv" and not drop_ls:
                continue
            res[key] = {}
            for ix in ["LSN", "LPI", "LPIish"]:
                c["x"] = np.log1p(c[ix])
                o, _, _, _ = fit(c, ["2010", "2011", "2016"], epoch_split_2010=True)
                W = SD.build(c, ix)
                r0, _ = SD.fit(W, ix, "none")
                r1, se1 = SD.fit(W, ix, "common", start=r0.x)
                res[key][ix] = {"tau": o["tau"], "ICC": o["ICC"], "n_rows": o["n_rows"],
                                "g_common": float(r1.x[10]), "g_common_se": se1.get(10),
                                "LR_common": float(2 * (r0.fun - r1.fun))}
                print(key, ix, res[key][ix], flush=True)
    (out / "site_terms_sensitivity.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
