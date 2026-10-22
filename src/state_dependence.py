"""Persistent site susceptibility versus state dependence (sequence memory).

Joint random-effects logit for sites with all three labels (Heckman-type treatment of
the initial condition, Gauss-Hermite integration):
    2010: logit p = a10[ep] + b10[ep] x10 + lam * d          (ep = P0 / later sounding)
    2011: logit p = a11 + b11 x11 + g11 * y10 + d
    2016: logit p = a16 + b16 x16 + g16 * y11 + d
    d ~ N(0, tau^2)
H0 (pure heterogeneity): g11 = g16 = 0.  Also reports a common-g model.
x = log(1 + LSN), log(1 + LPI) or log(1 + LPIish) for the event.

Usage: python -I state_dependence.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_expit
from scipy.stats import chi2

GH_X, GH_W = np.polynomial.hermite.hermgauss(40)
FEB11 = pd.Timestamp("2011-02-22")


def build(c, ix):
    w = c.pivot_table(index="rec_index", columns="event", values=["y", ix])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    ep = c[c["event"] == "2010"].set_index("rec_index")["p0"]
    w = w.join(ep.rename("p0")).dropna()
    return w


def nll(theta, W, ix, mode, return_parts=False):
    # theta: a10_0, a10_1, b10_0, b10_1, lam, a11, b11, a16, b16, log_tau, [g11, g16] or [g]
    a10 = theta[0:2]; b10 = theta[2:4]; lam = theta[4]
    a11, b11, a16, b16 = theta[5:9]
    tau = np.exp(theta[9])
    if mode == "none":
        g11 = g16 = 0.0
    elif mode == "common":
        g11 = g16 = theta[10]
    else:
        g11, g16 = theta[10], theta[11]
    ep = W["p0"].values.astype(int)
    x10, x11, x16 = (np.log1p(W[f"{ix}_{e}"].values) for e in ["2010", "2011", "2016"])
    y10, y11, y16 = (W[f"y_{e}"].values.astype(int) for e in ["2010", "2011", "2016"])
    d = np.sqrt(2.0) * tau * GH_X[None, :]
    e10 = (a10[ep] + b10[ep] * x10)[:, None] + lam * d
    e11 = (a11 + b11 * x11 + g11 * y10)[:, None] + d
    e16 = (a16 + b16 * x16 + g16 * y11)[:, None] + d
    ll = (np.where(y10[:, None] == 1, log_expit(e10), log_expit(-e10)) +
          np.where(y11[:, None] == 1, log_expit(e11), log_expit(-e11)) +
          np.where(y16[:, None] == 1, log_expit(e16), log_expit(-e16)))
    m = ll + np.log(GH_W / np.sqrt(np.pi))[None, :]
    mx = m.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(m - mx).sum(axis=1))
    return -lse.sum()


def fit(W, ix, mode, start=None):
    k = {"none": 10, "common": 11, "separate": 12}[mode]
    th0 = np.zeros(k)
    th0[2:4] = 1.0; th0[4] = 1.0; th0[6] = 1.0; th0[8] = 1.0
    if start is not None:
        n = min(len(start), k)
        th0[:n] = start[:n]
    # The likelihood has a ridge along which g trades off against tau and lam, and a single
    # L-BFGS-B start could stop early on it. As a precaution, fit from several starting values of g
    # and tau and keep the best (sd_start_check.py: all starts reach the same optimum).
    starts = [th0]
    for g0, lt0 in [(-0.8, np.log(1.0)), (0.8, np.log(1.0)), (0.8, np.log(2.0)), (0.0, np.log(2.0))]:
        th = th0.copy()
        th[9] = lt0
        if mode == "common":
            th[10] = g0
        elif mode == "separate":
            th[10] = g0
            th[11] = g0
        elif lt0 == np.log(1.0) and g0 != 0.0:
            continue
        starts.append(th)
    r = None
    for th in starts:
        ri = minimize(lambda t: nll(t, W, ix, mode), th, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7})
        if r is None or ri.fun < r.fun:
            r = ri
    # numerical Hessian for SEs of the last parameters
    h = 1e-4
    se = {}
    if mode != "none":
        idxs = list(range(10, k))
        H = np.zeros((k, k))
        f0 = r.fun
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ej = np.zeros(k); ei[i] = h; ej[j] = h
                fpp = nll(r.x + ei + ej, W, ix, mode); fpm = nll(r.x + ei - ej, W, ix, mode)
                fmp = nll(r.x - ei + ej, W, ix, mode); fmm = nll(r.x - ei - ej, W, ix, mode)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
        try:
            cov = np.linalg.inv(H)
            for i in idxs:
                se[i] = float(np.sqrt(max(cov[i, i], 0)))
        except np.linalg.LinAlgError:
            pass
    return r, se


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_parquet(out / "cases.parquet")
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    res = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        W = build(c, ix)
        r0, _ = fit(W, ix, "none")
        r1, se1 = fit(W, ix, "common", start=r0.x)
        r2, se2 = fit(W, ix, "separate", start=r0.x)
        row = {"n_sites": int(len(W)), "tau_H0": float(np.exp(r0.x[9])), "lam_H0": float(r0.x[4]),
               "tau_common": float(np.exp(r1.x[9])), "g_common": float(r1.x[10]), "g_common_se": se1.get(10),
               "OR_common": float(np.exp(r1.x[10])),
               "LR_common_vs_H0": float(2 * (r0.fun - r1.fun)), "p_common": float(chi2.sf(2 * (r0.fun - r1.fun), 1)),
               "tau_sep": float(np.exp(r2.x[9])), "g11": float(r2.x[10]), "g11_se": se2.get(10),
               "g16": float(r2.x[11]), "g16_se": se2.get(11),
               "LR_sep_vs_H0": float(2 * (r0.fun - r2.fun)), "p_sep": float(chi2.sf(2 * (r0.fun - r2.fun), 2))}
        res[ix] = row
        print(ix, json.dumps(row, indent=1, default=float), flush=True)
    (out / "state_dependence.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
