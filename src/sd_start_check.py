"""Does the common-g state-dependence model of state_dependence.py have local optima?

Fits the model from each starting value used by state_dependence.fit separately (no selection),
for one sample and index, and records the converged g, tau, lam and negative log-likelihood. The
first start (g0 = 0 from the H0 solution) is the single start used before the multi-start change.

Usage: python -I sd_start_check.py <outputs_dir> <excluding_ground_improved|all_sites> <index>
Writes <outputs_dir>/sd_start_check_<sample>_<index>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402

FILES = {"excluding_ground_improved": "cases.parquet", "all_sites": "cases_all.parquet"}


def main(out_dir, sample, ix):
    out = Path(out_dir)
    W = SD.build(base_cases(out, FILES[sample]), ix)
    r0, _ = SD.fit(W, ix, "none")
    th0 = np.zeros(11)
    th0[:10] = r0.x
    starts = [("H0 solution, g0=0 (former single start)", th0)]
    for g0, lt0 in [(-0.8, np.log(1.0)), (0.8, np.log(1.0)), (0.8, np.log(2.0)), (0.0, np.log(2.0))]:
        th = th0.copy()
        th[9] = lt0
        th[10] = g0
        starts.append((f"g0={g0:+.1f}, tau0={np.exp(lt0):.0f}", th))
    rows = []
    for label, th in starts:
        for opt_label, opts in [("default", {}), ("strict", {"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})]:
            r = minimize(lambda t: SD.nll(t, W, ix, "common"), th, method="L-BFGS-B", options=opts)
            rows.append({"start": label, "options": opt_label, "g": float(r.x[10]), "tau": float(np.exp(r.x[9])),
                         "lam": float(r.x[4]), "nll": float(r.fun), "converged": bool(r.success), "nit": int(r.nit)})
            print(rows[-1], flush=True)
    best = min(r["nll"] for r in rows)
    for r in rows:
        r["nll_minus_best"] = r["nll"] - best
    res = {"sample": sample, "index": ix, "n_sites": int(len(W)), "nll_H0": float(r0.fun), "fits": rows}
    (out / f"sd_start_check_{sample}_{ix}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
