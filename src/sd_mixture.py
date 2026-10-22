"""State dependence with a flexible (discrete-mixture) distribution of site terms.

The joint model of state_dependence.py assumes normal site terms. A misspecified heterogeneity
distribution biases state-dependence estimates, and in this database a cluster of sites with
manifestation in every event and low indices visibly moves the estimate. Following the
non-parametric approach of Heckman and Singer, the site term here takes K support points with
free locations and masses (mean fixed at zero):
    2010: logit p = a10[ep] + b10[ep] x10 + lam * d
    2011: logit p = a11 + b11 x11 + g * y10 + d
    2016: logit p = a16 + b16 x16 + g * y11 + d          d in {d_1..d_K} with masses pi_k
g is estimated and tested by likelihood ratio against g = 0 (same K). Several random starts are
used because mixture likelihoods are multimodal. Samples: all sites, and excluding the 122
ground-improved sites. K = 2, 3, 4; the normal model is reported alongside.

Usage: python -I sd_mixture.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_expit, logsumexp
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402

RNG = np.random.default_rng(2026)


def unpack(theta, K, with_g):
    a10, b10, lam = theta[0:2], theta[2:4], theta[4]
    a11, b11, a16, b16 = theta[5:9]
    draw = theta[9:9 + K]
    eta = np.concatenate([theta[9 + K:9 + 2 * K - 1], [0.0]])
    g = theta[9 + 2 * K - 1] if with_g else 0.0
    logpi = eta - logsumexp(eta)
    d = draw - np.sum(np.exp(logpi) * draw)
    return a10, b10, lam, a11, b11, a16, b16, d, logpi, g


def nll(theta, D, K, with_g):
    a10, b10, lam, a11, b11, a16, b16, d, logpi, g = unpack(theta, K, with_g)
    ep, x10, x11, x16, y10, y11, y16 = D
    e10 = (a10[ep] + b10[ep] * x10)[:, None] + lam * d[None, :]
    e11 = (a11 + b11 * x11 + g * y10)[:, None] + d[None, :]
    e16 = (a16 + b16 * x16 + g * y11)[:, None] + d[None, :]
    ll = (np.where(y10[:, None] == 1, log_expit(e10), log_expit(-e10)) +
          np.where(y11[:, None] == 1, log_expit(e11), log_expit(-e11)) +
          np.where(y16[:, None] == 1, log_expit(e16), log_expit(-e16)))
    return -np.sum(logsumexp(ll + logpi[None, :], axis=1))


def data(W, ix):
    ep = W["p0"].values.astype(int)
    x10, x11, x16 = (np.log1p(W[f"{ix}_{e}"].values) for e in ["2010", "2011", "2016"])
    y10, y11, y16 = (W[f"y_{e}"].values.astype(int) for e in ["2010", "2011", "2016"])
    return ep, x10, x11, x16, y10, y11, y16


def fit_mix(D, K, with_g, base, n_starts=8):
    best = None
    for s in range(n_starts):
        th = np.concatenate([base, np.sort(RNG.normal(0, 1.5, K)), RNG.normal(0, 0.5, K - 1), [0.0] if with_g else []])
        r = minimize(nll, th, args=(D, K, with_g), method="L-BFGS-B", options={"maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def main(out_dir):
    out = Path(out_dir)
    c = base_cases(out, "cases_all.parquet")  # full table; the ground-improved soundings are flagged below
    gi = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "ground_improvement"]]
    c = c.merge(gi, on="rec_index", how="left")
    c["gi"] = c["ground_improvement"].fillna(False).astype(bool)
    res = {}
    for sample, cc in {"all_sites": c, "excluding_ground_improved": c[~c["gi"]]}.items():
        res[sample] = {}
        for ix in ["LSN", "LPI", "LPIish"]:
            W = SD.build(cc, ix)
            D = data(W, ix)
            r_norm0, _ = SD.fit(W, ix, "none")
            r_norm1, _ = SD.fit(W, ix, "common", start=r_norm0.x)
            row = {"n_sites": int(len(W)), "normal": {"g": float(r_norm1.x[10]), "LR": float(2 * (r_norm0.fun - r_norm1.fun)),
                                                      "p": float(chi2.sf(2 * (r_norm0.fun - r_norm1.fun), 1)), "nll": float(r_norm1.fun)}}
            base = r_norm0.x[:9]
            for K in [2, 3, 4]:
                r0 = fit_mix(D, K, False, base)
                r1 = fit_mix(D, K, True, base)
                lr = float(2 * (r0.fun - r1.fun))
                a10, b10, lam, a11, b11, a16, b16, d, logpi, g = unpack(r1.x, K, True)
                row[f"K{K}"] = {"g": float(g), "LR": lr, "p": float(chi2.sf(max(lr, 0.0), 1)), "nll_g": float(r1.fun), "nll_0": float(r0.fun),
                                "support": [float(v) for v in d], "mass": [float(v) for v in np.exp(logpi)], "lam": float(lam),
                                "AIC_g": float(2 * r1.fun + 2 * len(r1.x))}
            res[sample][ix] = row
            print(sample, ix, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("g", "LR", "p", "AIC_g")}) for k, v in row.items()}), flush=True)
    (out / "sd_mixture.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
