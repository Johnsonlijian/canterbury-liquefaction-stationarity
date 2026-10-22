"""Paper A sensitivity: exclude the 122 soundings at sites flagged in the NZGD as ground-improved,
then refit the three-event site-term model and the joint state-dependence model.

Usage: python -I paperA_gi.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperA_robustness import base_cases, site_and_sd  # noqa: E402


def main(out_dir):
    out = Path(out_dir)
    c = base_cases(out, "cases_all.parquet")  # full table; the ground-improved soundings are flagged below
    gi = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "ground_improvement"]]
    c = c.merge(gi, on="rec_index", how="left")
    n_before = c["rec_index"].nunique()
    c = c[~c["ground_improvement"].fillna(False).astype(bool)].copy()
    res = {"sites_before": int(n_before), "sites_after": int(c["rec_index"].nunique())}
    for ix in ["LSN", "LPI", "LPIish"]:
        res[ix] = site_and_sd(c, ix)
        print(ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[ix].items()}, flush=True)
    (out / "paperA_robustness_gi.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
