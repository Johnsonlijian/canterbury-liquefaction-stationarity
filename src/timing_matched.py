"""Spatially matched test of CPT-timing effects on case-history discrimination.

Question: do soundings tested *after* an intervening major earthquake represent the
ground that existed at an *earlier* event as well as soundings tested before it?

Design (no label fitting of the index):
  * Groups: CPT tested before 2011-02-22 ("early") vs after ("late").
  * Footprint matching: late soundings are kept only if they lie within r metres of at
    least one early sounding (r = 50, 100, 200, 400 m). Discrimination (AUC) of the
    frozen index is then compared on the same footprint, so the spatial domain, the
    event demand field and the geology are shared.
  * Stratified concordance: AUC computed only over positive/negative pairs that lie
    in the same 500 m grid cell (removes between-area contrast entirely).
  * Uncertainty: 1-km spatial-block bootstrap (2,000 draws).

Usage: python -I timing_matched.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20261011)
INDICES = ["LSN", "LPI", "LPIish"]


def auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def stratified_auc(y, s, strata):
    """Concordance over (pos, neg) pairs within the same stratum."""
    num = 0.0
    den = 0.0
    df = pd.DataFrame({"y": y, "s": s, "g": strata})
    for _, g in df.groupby("g"):
        pos = g.loc[g["y"] == 1, "s"].values
        neg = g.loc[g["y"] == 0, "s"].values
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        num += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        den += float(diff.size)
    return num / den if den > 0 else np.nan, den


def block_ids(E, N, size=1000.0):
    return (np.floor(E / size).astype(int) * 100000 + np.floor(N / size).astype(int))


def boot_diff(df, idx, n_boot=2000):
    """Spatial-block bootstrap of AUC(early) - AUC(late) on a shared footprint."""
    blocks = df["block"].unique()
    b2rows = {b: np.where(df["block"].values == b)[0] for b in blocks}
    out = []
    for _ in range(n_boot):
        pick = RNG.choice(blocks, size=len(blocks), replace=True)
        rows = np.concatenate([b2rows[b] for b in pick])
        d = df.iloc[rows]
        e = d[d["early"] == 1]
        l = d[d["early"] == 0]
        a_e = auc(e["y"].values, e[idx].values)
        a_l = auc(l["y"].values, l[idx].values)
        if np.isfinite(a_e) and np.isfinite(a_l):
            out.append(a_e - a_l)
    out = np.array(out)
    return float(np.mean(out)), [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))], float(np.mean(out <= 0))


def analyse_event(cases, ev, variant="inv"):
    d = cases[(cases["variant"] == variant) & (cases["event"] == ev) & cases["y"].notna()].copy()
    d["early"] = (d["date"] < pd.Timestamp("2011-02-22")).astype(int)
    d = d[d["date"].notna()]
    d["block"] = block_ids(d["E"].values, d["N"].values, 1000.0)
    d["cell500"] = block_ids(d["E"].values, d["N"].values, 500.0)
    early = d[d["early"] == 1]
    tree = cKDTree(early[["E", "N"]].values)
    dist, _ = tree.query(d[["E", "N"]].values, k=1)
    d["dist_to_early"] = dist
    res = {"event": ev, "variant": variant, "n_early": int(len(early)), "n_late_all": int((d["early"] == 0).sum()),
           "base_rate_early": float(early["y"].mean()), "base_rate_late_all": float(d.loc[d["early"] == 0, "y"].mean()),
           "median_PGA_early": float(early["PGA"].median()), "median_PGA_late_all": float(d.loc[d["early"] == 0, "PGA"].median())}
    for idx in INDICES:
        res[f"AUC_{idx}_early"] = auc(early["y"].values, early[idx].values)
        res[f"AUC_{idx}_late_all"] = auc(d.loc[d["early"] == 0, "y"].values, d.loc[d["early"] == 0, idx].values)
    foot = []
    for r in [50, 100, 200, 400]:
        sub = d[(d["early"] == 1) | (d["dist_to_early"] <= r)]
        late = sub[sub["early"] == 0]
        row = {"radius_m": r, "n_late": int(len(late)), "base_rate_late": float(late["y"].mean()),
               "median_PGA_late": float(late["PGA"].median())}
        for idx in INDICES:
            row[f"AUC_{idx}_late"] = auc(late["y"].values, late[idx].values)
            mean, ci, p = boot_diff(sub, idx, n_boot=1000 if r < 400 else 2000)
            row[f"dAUC_{idx}_early_minus_late"] = mean
            row[f"dAUC_{idx}_ci95"] = ci
            row[f"dAUC_{idx}_boot_p_le0"] = p
        foot.append(row)
    res["footprint"] = foot
    # within-cell concordance (500 m cells), restricted to cells that contain both groups
    both = d.groupby("cell500")["early"].agg(["min", "max"])
    cells = both[(both["min"] == 0) & (both["max"] == 1)].index
    dc = d[d["cell500"].isin(cells)]
    strat = {"n_cells_with_both": int(len(cells)), "n_early": int((dc["early"] == 1).sum()), "n_late": int((dc["early"] == 0).sum())}
    for idx in INDICES:
        for grp, name in [(1, "early"), (0, "late")]:
            g = dc[dc["early"] == grp]
            a, npairs = stratified_auc(g["y"].values, g[idx].values, g["cell500"].values)
            strat[f"withincell_AUC_{idx}_{name}"] = a
            strat[f"withincell_pairs_{idx}_{name}"] = npairs
    res["within_cell500"] = strat
    return res, d


def main(out_dir):
    out = Path(out_dir)
    cases = pd.read_parquet(out / "cases.parquet")
    results = {}
    for ev in ["2010", "2011"]:
        for variant in ["inv", "meas"]:
            r, d = analyse_event(cases, ev, variant)
            results[f"{ev}_{variant}"] = r
            print(json.dumps(r, indent=1, default=float)[:6000])
            print("-" * 60, flush=True)
    (out / "timing_matched.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
