"""State-dependence model of state_dependence.py for all sites, including the 122 ground-improved
soundings (cases_all.parquet), for the sensitivity table of Paper A.

Same multi-start fit and numerical-Hessian standard error as the other rows of that table (H0 and
common-g models only). One index per call so that the indices can run in parallel.

Usage: python -I sd_allsites.py <outputs_dir> <LSN|LPI|LPIish>
Writes <outputs_dir>/sd_allsites_<index>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402


def main(out_dir, ix):
    out = Path(out_dir)
    c = base_cases(out, "cases_all.parquet")
    W = SD.build(c, ix)
    r0, _ = SD.fit(W, ix, "none")
    r1, se1 = SD.fit(W, ix, "common", start=r0.x)
    lr = 2 * (r0.fun - r1.fun)
    row = {"sample": "all_sites", "index": ix, "n_sites": int(len(W)),
           "tau_H0": float(np.exp(r0.x[9])), "lam_H0": float(r0.x[4]),
           "tau_common": float(np.exp(r1.x[9])), "g_common": float(r1.x[10]), "g_common_se": se1.get(10),
           "OR_common": float(np.exp(r1.x[10])), "LR_common_vs_H0": float(lr), "p_common": float(chi2.sf(max(lr, 0.0), 1))}
    print(ix, json.dumps(row, default=float), flush=True)
    (out / f"sd_allsites_{ix}.json").write_text(json.dumps(row, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
