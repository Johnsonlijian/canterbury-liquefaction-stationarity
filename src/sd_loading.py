"""Is state dependence identified when the site term may load differently on each earthquake?

The joint model of state_dependence.py lets the site term load on 2010 through lam but fixes its
loading on 2011 and 2016 to one. The forecast analysis shows that the 2016 event needs a smaller
weight on the site term, so that restriction may not hold. Here the profile log-likelihood of a
common state-dependence coefficient g is computed on a grid, under
  (a) the original restriction (loading on 2016 = 1), and
  (b) a free loading on 2016 (lam16),
for all sites and excluding ground-improved sites. A flat profile under (b) means that the data
cannot separate state dependence from event-dependent site effects.

Usage: python -I sd_loading.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402

GH_X, GH_W = np.polynomial.hermite.hermgauss(30)
GRID = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]


def nll(theta, D, g, free16):
    a10, b10, lam = theta[0:2], theta[2:4], theta[4]
    a11, b11, a16, b16 = theta[5:9]
    tau = np.exp(theta[9])
    lam16 = theta[10] if free16 else 1.0
    ep, x10, x11, x16, y10, y11, y16 = D
    d = np.sqrt(2.0) * tau * GH_X[None, :]
    e10 = (a10[ep] + b10[ep] * x10)[:, None] + lam * d
    e11 = (a11 + b11 * x11 + g * y10)[:, None] + d
    e16 = (a16 + b16 * x16 + g * y11)[:, None] + lam16 * d
    ll = (np.where(y10[:, None] == 1, log_expit(e10), log_expit(-e10)) + np.where(y11[:, None] == 1, log_expit(e11), log_expit(-e11)) +
          np.where(y16[:, None] == 1, log_expit(e16), log_expit(-e16)))
    m = ll + np.log(GH_W / np.sqrt(np.pi))[None, :]
    mx = m.max(axis=1, keepdims=True)
    return -np.sum(mx[:, 0] + np.log(np.exp(m - mx).sum(axis=1)))


def main(out_dir):
    out = Path(out_dir)
    c = base_cases(out, "cases_all.parquet")  # full table; the ground-improved soundings are flagged below
    gi = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "ground_improvement"]]
    c = c.merge(gi, on="rec_index", how="left")
    c["gi"] = c["ground_improvement"].fillna(False).astype(bool)
    res = {}
    for sample, cc in {"all_sites": c, "excluding_ground_improved": c[~c["gi"]]}.items():
        res[sample] = {}
        for ix in ["LSN", "LPIish"]:
            W = SD.build(cc, ix)
            ep = W["p0"].values.astype(int)
            x = [np.log1p(W[f"{ix}_{e}"].values) for e in ["2010", "2011", "2016"]]
            y = [W[f"y_{e}"].values.astype(int) for e in ["2010", "2011", "2016"]]
            D = (ep, x[0], x[1], x[2], y[0], y[1], y[2])
            r0, _ = SD.fit(W, ix, "none")
            out_ix = {}
            for free16 in [False, True]:
                prof = []
                start = np.concatenate([r0.x[:10], [1.0]]) if free16 else r0.x[:10].copy()
                for g in GRID:
                    r = minimize(nll, start, args=(D, g, free16), method="L-BFGS-B")
                    prof.append({"g": g, "nll": float(r.fun), "tau": float(np.exp(r.x[9])), "lam16": float(r.x[10]) if free16 else 1.0})
                    start = r.x
                best = min(p["nll"] for p in prof)
                for p in prof:
                    p["dev"] = 2 * (p["nll"] - best)
                out_ix["free_lam16" if free16 else "lam16_fixed_1"] = prof
                print(sample, ix, "free16" if free16 else "fixed", [(p["g"], round(p["dev"], 2), round(p["lam16"], 2)) for p in prof], flush=True)
            res[sample][ix] = out_ix
    (out / "sd_loading.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
