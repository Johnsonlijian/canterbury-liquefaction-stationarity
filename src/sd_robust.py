"""Robust (multi-start) estimation of the joint state-dependence model of state_dependence.py.

The likelihood has a ridge along which g trades off against tau and lam, so as a precaution every
model is fitted from a grid of starting values (tau0 in {0.8, 1.4, 2.2} for H0; g0 in {-1, 0, 1}
for the common model and {-0.5, 0.5} for the separate model, each with tau0 in {1, 2}; lam0 = 1),
the best solution is polished with Powell, and a profile-likelihood interval for g is computed on
a 0.2-step grid from -0.8 to 1.8 with warm starts from the optimum. (sd_start_check.py found that
all starts reach the same optimum, so this is a safeguard rather than a correction.) Models: H0
(g = 0), common g, separate g11 and g16. Samples: all sites and excluding the 122 ground-improved
sites; indices LSN, LPI, LPIish. 40-node Gauss-Hermite.

Usage: python -I sd_robust.py <outputs_dir>                    (all samples and indices, sequential)
       python -I sd_robust.py <outputs_dir> <sample> <index>    (one part, for parallel runs)
       python -I sd_robust.py <outputs_dir> merge               (assemble sd_robust.json from the parts)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402


def best_fit(W, ix, mode, base):
    k = {"none": 10, "common": 11, "separate": 12}[mode]
    starts = []
    for g0, lt0, l0 in itertools.product(([-1.0, 0.0, 1.0] if mode == "common" else [-0.5, 0.5]) if mode != "none" else [0.0], [np.log(1.0), np.log(2.0)] if mode != "none" else [np.log(0.8), np.log(1.4), np.log(2.2)], [1.0]):
        th = np.zeros(k)
        th[:9] = base[:9]
        th[4] = l0
        th[9] = lt0
        if mode == "common":
            th[10] = g0
        elif mode == "separate":
            th[10] = g0
            th[11] = g0
        starts.append(th)
    best = None
    for th in starts:
        r = minimize(lambda t: SD.nll(t, W, ix, mode), th, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    pol = minimize(lambda t: SD.nll(t, W, ix, mode), best.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-10})
    return pol if pol.fun < best.fun else best


def profile_g(W, ix, r1, grid):
    out = []
    for g in grid:
        f = lambda t: SD.nll(np.concatenate([t, [g]]), W, ix, "common")
        r = minimize(f, r1.x[:10], method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12})
        out.append((g, float(r.fun)))
    return out


# cases.parquet excludes the ground-improved soundings; cases_all.parquet keeps them
SAMPLES = {"excluding_ground_improved": "cases.parquet", "all_sites": "cases_all.parquet"}
INDICES = ["LSN", "LPI", "LPIish"]


def run_one(cc, ix):
    W = SD.build(cc, ix)
    r_init, _ = SD.fit(W, ix, "none")
    r0 = best_fit(W, ix, "none", r_init.x)
    r1 = best_fit(W, ix, "common", r0.x)
    r2 = best_fit(W, ix, "separate", r0.x)
    grid = np.round(np.arange(-0.8, 1.81, 0.2), 2)
    prof = profile_g(W, ix, r1, grid)
    nll_min = min(r1.fun, min(p[1] for p in prof))
    dev = np.array([2 * (p[1] - nll_min) for p in prof])
    inside = grid[dev <= chi2.ppf(0.95, 1)]
    return {"n_sites": int(len(W)), "nll_H0": float(r0.fun), "tau_H0": float(np.exp(r0.x[9])), "lam_H0": float(r0.x[4]),
            "nll_common": float(r1.fun), "g_common": float(r1.x[10]), "tau_common": float(np.exp(r1.x[9])), "lam_common": float(r1.x[4]),
            "LR_common": float(2 * (r0.fun - r1.fun)), "p_common": float(chi2.sf(max(2 * (r0.fun - r1.fun), 0), 1)),
            "g_profile_ci95": [float(inside.min()), float(inside.max())] if inside.size else None,
            "profile": [[float(g), float(d)] for g, d in zip(grid, dev)],
            "nll_sep": float(r2.fun), "g11": float(r2.x[10]), "g16": float(r2.x[11]), "tau_sep": float(np.exp(r2.x[9])),
            "LR_sep": float(2 * (r0.fun - r2.fun)), "p_sep": float(chi2.sf(max(2 * (r0.fun - r2.fun), 0), 2))}


def show(sample, ix, row):
    print(sample, ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "profile"}, flush=True)


def main(out_dir):
    out = Path(out_dir)
    res = {}
    for sample, fname in SAMPLES.items():
        cc = base_cases(out, fname)
        res[sample] = {}
        for ix in INDICES:
            res[sample][ix] = run_one(cc, ix)
            show(sample, ix, res[sample][ix])
    (out / "sd_robust.json").write_text(json.dumps(res, indent=1, default=float))


def part(out_dir, sample, ix):
    out = Path(out_dir)
    row = run_one(base_cases(out, SAMPLES[sample]), ix)
    show(sample, ix, row)
    (out / f"sd_robust_part_{sample}_{ix}.json").write_text(json.dumps(row, indent=1, default=float))


def merge(out_dir):
    out = Path(out_dir)
    res = {}
    for sample in SAMPLES:
        for ix in INDICES:
            f = out / f"sd_robust_part_{sample}_{ix}.json"
            if f.exists():
                res.setdefault(sample, {})[ix] = json.loads(f.read_text())
    (out / "sd_robust.json").write_text(json.dumps(res, indent=1, default=float))
    print({s: sorted(v) for s, v in res.items()})


if __name__ == "__main__":
    if len(sys.argv) == 4:
        part(*sys.argv[1:4])
    elif len(sys.argv) == 3 and sys.argv[2] == "merge":
        merge(sys.argv[1])
    else:
        main(sys.argv[1])
