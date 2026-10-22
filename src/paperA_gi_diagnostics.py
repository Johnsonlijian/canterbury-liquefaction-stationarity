"""Why do the 122 ground-improved sites matter for state dependence?

1. Profile of the flagged sites: test dates, programmes, label histories, indices.
2. State-dependence model without them: common and event-specific coefficients (2010->2011,
   2011->2016), and the naive conditional association, for each index.
3. Influence check: refit the common-coefficient model after dropping random subsets of 122
   non-flagged sites (10 draws), to see whether any 122 sites move the estimate this much.

Usage: python -I paperA_gi_diagnostics.py <outputs_dir>                 (everything)
       python -I paperA_gi_diagnostics.py <outputs_dir> draws <k0> <k1>  (refit draws k0..k1-1)
       python -I paperA_gi_diagnostics.py <outputs_dir> merge            (put refitted draws in the JSON)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_dependence as SD  # noqa: E402
from paperA_robustness import base_cases  # noqa: E402


def sd_all(c, ix):
    W = SD.build(c, ix)
    r0, _ = SD.fit(W, ix, "none")
    r1, se1 = SD.fit(W, ix, "common", start=r0.x)
    r2, se2 = SD.fit(W, ix, "separate", start=r0.x)
    return {"n_sites": int(len(W)), "tau_H0": float(np.exp(r0.x[9])), "g_common": float(r1.x[10]), "g_common_se": se1.get(10),
            "p_common": float(chi2.sf(2 * (r0.fun - r1.fun), 1)), "g11": float(r2.x[10]), "g11_se": se2.get(10),
            "g16": float(r2.x[11]), "g16_se": se2.get(11), "p_sep": float(chi2.sf(2 * (r0.fun - r2.fun), 2)),
            "tau_sep": float(np.exp(r2.x[9]))}


def load(out):
    c = base_cases(out, "cases_all.parquet")  # full table; the ground-improved soundings are flagged below
    pc = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "ground_improvement", "programme", "period"]]
    c = c.merge(pc, on="rec_index", how="left")
    c["gi"] = c["ground_improvement"].fillna(False).astype(bool)
    return c


def subsets(c, n=10):
    # the same sequence of random subsets as the loop in main()
    rng = np.random.default_rng(122)
    sites = c.loc[~c["gi"], "rec_index"].unique()
    return [rng.choice(sites, size=122, replace=False) for _ in range(n)]


def draw_g(c, drop, ix="LSN"):
    W = SD.build(c[~c["rec_index"].isin(drop)], ix)
    r0, _ = SD.fit(W, ix, "none")
    r1, _ = SD.fit(W, ix, "common", start=r0.x)
    return float(r1.x[10])


def draws_part(out_dir, k0, k1):
    """Refit draws k0..k1-1 of the influence check (for parallel runs)."""
    out = Path(out_dir)
    c = load(out)
    sub = subsets(c)
    res = {"k": [], "g": []}
    for k in range(k0, k1):
        res["k"].append(k)
        res["g"].append(draw_g(c, sub[k]))
        print("random drop", k, round(res["g"][-1], 3), flush=True)
    (out / f"paperA_gi_draws_{k0}_{k1}.json").write_text(json.dumps(res, indent=1))


def merge_draws(out_dir):
    """Replace the influence-check draws in paperA_gi_diagnostics.json by the refitted parts."""
    out = Path(out_dir)
    f = out / "paperA_gi_diagnostics.json"
    res = json.loads(f.read_text())
    g = {}
    for p in sorted(out.glob("paperA_gi_draws_*.json")):
        d = json.loads(p.read_text())
        g.update(dict(zip(d["k"], d["g"])))
    if sorted(g) != list(range(10)):
        raise SystemExit(f"draws missing: have {sorted(g)}")
    if "random_drop_122_LSN_g_common_single_start" not in res:
        res["random_drop_122_LSN_g_common_single_start"] = res["random_drop_122_LSN_g_common"]
    res["random_drop_122_LSN_g_common"] = [g[k] for k in range(10)]
    res["_note"] = ("random_drop_122_LSN_g_common: multi-start fit of state_dependence.fit (same subsets as the first "
                    "run, whose single-start values are kept for comparison). sd_excluding_gi comes from the first run "
                    "(single start); it agrees with the multi-start state_dependence.json to within 0.001 in g_common.")
    f.write_text(json.dumps(res, indent=1, default=float))
    print(res["random_drop_122_LSN_g_common"])


def main(out_dir):
    out = Path(out_dir)
    c = load(out)
    res = {}
    g = c[c["gi"]]
    wide = g.pivot_table(index="rec_index", columns="event", values="y")
    res["gi_profile"] = {"n_sites": int(g["rec_index"].nunique()),
                         "periods": g.drop_duplicates("rec_index")["period"].astype(str).value_counts().to_dict(),
                         "programmes": g.drop_duplicates("rec_index")["programme"].value_counts().to_dict(),
                         "date_range": [str(g["date"].min().date()), str(g["date"].max().date())],
                         "manifest_rate_by_event": wide.mean().round(3).to_dict(),
                         "n_with_label_by_event": wide.notna().sum().astype(int).to_dict(),
                         "median_LSN_by_event": g.groupby("event")["LSN"].median().round(2).to_dict(),
                         "median_LSN_by_event_others": c[~c["gi"]].groupby("event")["LSN"].median().round(2).to_dict(),
                         "history_2011_2016": {f"{int(a)}{int(b)}": int(n) for (a, b), n in wide.dropna(subset=["2011", "2016"]).groupby(["2011", "2016"]).size().items()}}
    print(json.dumps(res["gi_profile"], indent=1))
    res["sd_excluding_gi"] = {}
    res["sd_all"] = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        res["sd_excluding_gi"][ix] = sd_all(c[~c["gi"]], ix)
        print("excl GI", ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["sd_excluding_gi"][ix].items()}, flush=True)
    # influence check with random subsets of non-flagged sites (LSN only)
    draws = []
    for k, drop in enumerate(subsets(c)):
        draws.append(draw_g(c, drop))
        print("random drop", k, round(draws[-1], 3), flush=True)
    res["random_drop_122_LSN_g_common"] = draws
    (out / "paperA_gi_diagnostics.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[2] == "draws":
        draws_part(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 3 and sys.argv[2] == "merge":
        merge_draws(sys.argv[1])
    else:
        main(sys.argv[1])
