"""Repeatable site terms in liquefaction manifestation across the Canterbury sequence.

Random-intercept logistic model (exact marginal likelihood, Gauss-Hermite quadrature):
    logit P(y_se = 1 | d_s) = a_e + b_e * x_se + d_s,    d_s ~ N(0, tau^2)
x_se = log(1 + index_se) with the index computed for event e (event PGA, Mw, GWT).
Nothing in the index is fitted; a_e, b_e, tau are fitted.

Outputs
  * tau, ICC = tau^2 / (tau^2 + pi^2/3), LR test against tau = 0 (boundary, chi-bar^2).
  * Leave-one-event-out (LOEO) forecast value: rank held-out event h by the index alone
    ("ergodic") versus b_bar * x_sh + E[d_s | other events] ("non-ergodic"); AUC, 1-km
    spatial-block bootstrap of the AUC gain. Also a prior-label-only comparator.
  * Empirical-Bayes site terms (all events) for spatial analysis and mapping.
  * Epoch control: the 2010 equation gets separate (a, b) for soundings tested before
    22 Feb 2011 (P0) and after (label-resolution/provenance effect, see timing analysis).

Usage: python -I site_terms_glmm.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from sklearn.metrics import roc_auc_score

GH_X, GH_W = np.polynomial.hermite.hermgauss(40)
RNG = np.random.default_rng(20261011)
FEB11 = pd.Timestamp("2011-02-22")


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def design(df, events, epoch_split_2010=True):
    """Return per-row eq-id, x, y, site codes. Equation ids index (a, b) pairs."""
    eq_names = []
    eq_id = np.empty(len(df), int)
    for k, (ev, ep) in enumerate(zip(df["event"].values, df["p0"].values)):
        key = f"{ev}_P0" if (ev == "2010" and epoch_split_2010 and ep == 1) else ev
        if key not in eq_names:
            eq_names.append(key)
        eq_id[k] = eq_names.index(key)
    sites, site_idx = np.unique(df["rec_index"].values, return_inverse=True)
    return eq_names, eq_id, site_idx, len(sites)


def negloglik(theta, x, y, eq_id, site_idx, n_sites, n_eq, return_post=False):
    a = theta[:n_eq]
    b = theta[n_eq:2 * n_eq]
    tau = np.exp(theta[-1])
    eta0 = a[eq_id] + b[eq_id] * x                      # (n,)
    d = np.sqrt(2.0) * tau * GH_X                       # (K,)
    eta = eta0[:, None] + d[None, :]                    # (n, K)
    ll = np.where(y[:, None] == 1, log_expit(eta), log_expit(-eta))
    site_ll = np.zeros((n_sites, len(GH_X)))
    np.add.at(site_ll, site_idx, ll)
    lw = np.log(GH_W / np.sqrt(np.pi))
    m = site_ll + lw[None, :]
    mx = m.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(m - mx).sum(axis=1))
    if return_post:
        post_w = np.exp(m - lse[:, None])
        mean = post_w @ d
        var = post_w @ d ** 2 - mean ** 2
        return -lse.sum(), mean, var
    return -lse.sum()


def fit(df, events, epoch_split_2010=True, x_col="x"):
    sub = df[df["event"].isin(events)].copy()
    eq_names, eq_id, site_idx, n_sites = design(sub, events, epoch_split_2010)
    x = sub[x_col].values.astype(float)
    y = sub["y"].values.astype(int)
    n_eq = len(eq_names)
    theta0 = np.concatenate([np.zeros(n_eq), np.ones(n_eq), [0.0]])
    f = lambda t: negloglik(t, x, y, eq_id, site_idx, n_sites, n_eq)
    r = minimize(f, theta0, method="L-BFGS-B")
    # tau = 0 comparison (ordinary logistic with the same equations)
    t0 = np.concatenate([r.x[:2 * n_eq], [-12.0]])
    f0 = lambda t: negloglik(np.concatenate([t, [-12.0]]), x, y, eq_id, site_idx, n_sites, n_eq)
    r0 = minimize(f0, t0[:-1], method="L-BFGS-B")
    _, post_mean, post_var = negloglik(r.x, x, y, eq_id, site_idx, n_sites, n_eq, return_post=True)
    sites = np.unique(sub["rec_index"].values)
    tau = float(np.exp(r.x[-1]))
    lr = float(2 * (r0.fun - r.fun))
    out = {"events": events, "eq_names": eq_names, "n_rows": int(len(sub)), "n_sites": int(n_sites),
           "a": dict(zip(eq_names, map(float, r.x[:n_eq]))), "b": dict(zip(eq_names, map(float, r.x[n_eq:2 * n_eq]))),
           "tau": tau, "ICC": tau ** 2 / (tau ** 2 + np.pi ** 2 / 3), "LR_vs_tau0": lr,
           "p_chibar2": float(0.5 * __import__("scipy").stats.chi2.sf(lr, 1)), "converged": bool(r.success)}
    eb = pd.DataFrame({"rec_index": sites, "d_mean": post_mean, "d_var": post_var})
    return out, eb, r.x, eq_names


def eb_from_params(df_other, theta, eq_names, x_col="x"):
    """Posterior mean site term from a site's other-event outcomes under fitted params."""
    n_eq = len(eq_names)
    eq_id = np.array([eq_names.index(f"{ev}_P0" if (ev == "2010" and p0 == 1 and f"{ev}_P0" in eq_names) else ev)
                      for ev, p0 in zip(df_other["event"].values, df_other["p0"].values)])
    sites, site_idx = np.unique(df_other["rec_index"].values, return_inverse=True)
    _, mean, var = negloglik(theta, df_other[x_col].values.astype(float), df_other["y"].values.astype(int),
                             eq_id, site_idx, len(sites), n_eq, return_post=True)
    return pd.DataFrame({"rec_index": sites, "d_hat": mean, "d_var": var})


def block_boot_gain(d, s0, s1, n_boot=2000):
    blocks = d["block"].values
    ub = np.unique(blocks)
    rows = {b: np.where(blocks == b)[0] for b in ub}
    gains = []
    for _ in range(n_boot):
        pick = RNG.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([rows[b] for b in pick])
        yy = d["y"].values[idx]
        if len(np.unique(yy)) < 2:
            continue
        gains.append(auc(yy, d[s1].values[idx]) - auc(yy, d[s0].values[idx]))
    g = np.array(gains)
    return float(np.mean(g)), [float(np.quantile(g, .025)), float(np.quantile(g, .975))], float(np.mean(g <= 0))


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_parquet(out / "cases.parquet")
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    c["block"] = (np.floor(c["E"] / 1000).astype(int) * 100000 + np.floor(c["N"] / 1000).astype(int))
    res = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        c["x"] = np.log1p(c[ix])
        R = {}
        for evs in [["2011", "2016"], ["2010", "2011", "2016"]]:
            o, eb, theta, names = fit(c, evs, epoch_split_2010=True)
            R["+".join(evs)] = o
            if len(evs) == 3:
                eb.to_parquet(out / f"site_terms_eb_{ix}.parquet", index=False)
        # 2010 without epoch split (shows provenance effect on tau)
        o, _, _, _ = fit(c, ["2010", "2011", "2016"], epoch_split_2010=False)
        R["2010+2011+2016_noEpochSplit"] = o

        # LOEO forecast value
        loeo = []
        all_ev = ["2010", "2011", "2016"]
        for h in all_ev:
            train_ev = [e for e in all_ev if e != h]
            o, eb, theta, names = fit(c, train_ev, epoch_split_2010=True)
            other = c[c["event"].isin(train_ev)]
            ebh = eb_from_params(other, theta, names)
            d = c[c["event"] == h].merge(ebh, on="rec_index", how="inner")
            bbar = float(np.mean([o["b"][k] for k in names]))
            d["s_ergodic"] = d["x"]
            d["s_nonergodic"] = bbar * d["x"] + d["d_hat"]
            # prior-label-only comparator: count of manifested other events (no CPT)
            lab = other.pivot_table(index="rec_index", columns="event", values="y")
            d = d.merge(lab.sum(axis=1).rename("n_prior_manifest"), left_on="rec_index", right_index=True, how="left")
            row = {"held_out": h, "n": int(len(d)), "base_rate": float(d["y"].mean()),
                   "AUC_ergodic_index": auc(d["y"], d["s_ergodic"]),
                   "AUC_nonergodic": auc(d["y"], d["s_nonergodic"]),
                   "AUC_prior_labels_only": auc(d["y"], d["n_prior_manifest"]),
                   "tau_train": o["tau"]}
            g, ci, p = block_boot_gain(d, "s_ergodic", "s_nonergodic", n_boot=1000)
            row.update({"AUC_gain": g, "AUC_gain_ci95": ci, "boot_p_gain_le0": p})
            if h == "2010":
                for ep, gg in d.groupby("p0"):
                    row[f"AUC_ergodic_p0={ep}"] = auc(gg["y"], gg["s_ergodic"])
                    row[f"AUC_nonergodic_p0={ep}"] = auc(gg["y"], gg["s_nonergodic"])
            loeo.append(row)
        R["LOEO"] = loeo
        res[ix] = R
        print(ix, json.dumps(R, indent=1, default=float)[:5000], flush=True)
    (out / "site_terms_glmm.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
