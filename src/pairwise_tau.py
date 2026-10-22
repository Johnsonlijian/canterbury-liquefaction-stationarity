"""Repeatable site variance for each pair of earthquakes, and a sequential forecast of 2016.

1. For each pair of events (2010-2011, 2011-2016, 2010-2016) the random-intercept model of
   site_terms_glmm.py is fitted, with a profile-likelihood interval for tau on a grid and the
   boundary likelihood-ratio test of tau = 0. Samples: the primary sample (ground-improved sites
   excluded; cases.parquet), all sites (cases_all.parquet; LSN only), and sites sounded before
   22 February 2011 (site-level 2010 labels; LSN only).
2. Sequential forecast of 2016: the model is fitted to 2010 and 2011 (the past), and each site's
   term is predicted from its 2011 outcome only, from its 2010 outcome only, or from both; the
   2016 AUC of b_bar * x + w * d_hat is compared with the index alone (1-km block bootstrap).

Usage: python -I pairwise_tau.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_terms_glmm import auc, block_boot_gain, design, eb_from_params, fit, negloglik  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")
PAIRS = [("2010", "2011"), ("2011", "2016"), ("2010", "2016")]
TAU_GRID = [0.001, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def load(out, fname):
    c = pd.read_parquet(out / fname)
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    c["block"] = (np.floor(c["E"] / 1000).astype(int) * 100000 + np.floor(c["N"] / 1000).astype(int))
    return c


def profile_tau(c, events, theta_hat):
    sub = c[c["event"].isin(events)]
    names, eq_id, site_idx, n_sites = design(sub, events, True)
    x, y = sub["x"].values.astype(float), sub["y"].values.astype(int)
    n_eq = len(names)
    best = None
    prof = []
    start = theta_hat[:2 * n_eq]
    for t in TAU_GRID:
        f = lambda th: negloglik(np.concatenate([th, [np.log(t)]]), x, y, eq_id, site_idx, n_sites, n_eq)
        r = minimize(f, start, method="L-BFGS-B")
        prof.append((t, float(r.fun)))
        start = r.x
    return prof


def pair_table(c, ix, pairs):
    c = c.copy()
    c["x"] = np.log1p(c[ix])
    res = {}
    for ev in pairs:
        o, _, theta, _ = fit(c, list(ev), epoch_split_2010=True)
        prof = profile_tau(c, list(ev), theta)
        nll_hat = min(min(p[1] for p in prof), -1e300 if False else min(p[1] for p in prof))
        o_nll = None
        dev = [(t, 2 * (v - nll_hat)) for t, v in prof]
        inside = [t for t, d in dev if d <= chi2.ppf(0.95, 1)]
        res["-".join(ev)] = {"tau": o["tau"], "ICC": o["ICC"], "LR_vs_tau0": o["LR_vs_tau0"], "p": o["p_chibar2"],
                             "tau_ci95_grid": [min(inside), max(inside)] if inside else None, "profile": dev,
                             "n_sites": o["n_sites"], "n_rows": o["n_rows"]}
        print(ix, ev, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["-".join(ev)].items() if k != "profile"}, flush=True)
    return res


def sequential_2016(c, ix):
    c = c.copy()
    c["x"] = np.log1p(c[ix])
    o, _, theta, names = fit(c, ["2010", "2011"], epoch_split_2010=True)
    bbar = float(np.mean([o["b"][k] for k in names]))
    d = c[c["event"] == "2016"].copy()
    d["s_erg"] = d["x"]
    out = {"tau_2010_2011": o["tau"]}
    for tag, evs in {"from_2011": ["2011"], "from_2010": ["2010"], "from_both": ["2010", "2011"]}.items():
        eb = eb_from_params(c[c["event"].isin(evs)], theta, names)
        dd = d.merge(eb[["rec_index", "d_hat"]], on="rec_index", how="left").fillna({"d_hat": 0.0})
        for w in [1.0, 0.5, 0.25]:
            dd[f"s_{w}"] = bbar * dd["x"] + w * dd["d_hat"]
            g, ci, _ = block_boot_gain(dd, "s_erg", f"s_{w}", n_boot=1000)
            out[f"{tag}_w{w}"] = {"AUC": auc(dd["y"], dd[f"s_{w}"]), "gain": g, "ci": ci}
    out["AUC_index"] = auc(d["y"], d["s_erg"])
    print(ix, "sequential 2016", {k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in out.items()}, flush=True)
    return out


def main(out_dir):
    out = Path(out_dir)
    prim = load(out, "cases.parquet")
    allc = load(out, "cases_all.parquet")
    res = {"primary": {}, "all_sites": {}, "p0_sites": {}, "sequential_2016": {}}
    for ix in ["LSN", "LPI", "LPIish"]:
        res["primary"][ix] = pair_table(prim, ix, PAIRS)
        res["sequential_2016"][ix] = sequential_2016(prim, ix)
    res["all_sites"]["LSN"] = pair_table(allc, "LSN", PAIRS)
    res["p0_sites"]["LSN"] = pair_table(prim[prim["p0"] == 1], "LSN", PAIRS)
    (out / "pairwise_tau.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
