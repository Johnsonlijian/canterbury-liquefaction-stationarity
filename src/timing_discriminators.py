"""Discriminate 'ground-state change' from 'purposeful siting' for the CPT-timing effect.

Tests
  T1  2016 outcome on the shared footprint: state change predicts that late (post-2011)
      soundings rank 2016 manifestation at least as well as early ones; siting on 2010
      damage predicts an early advantage (or parity) because 2010 and 2016 susceptibility
      co-vary.
  T2  Dose-response for the 2010 outcome: late soundings split by the number of
      intervening liquefaction-producing events (Feb 2011 only: tested 22 Feb-12 Jun 2011;
      Feb+Jun: 13 Jun-22 Dec 2011; Feb+Jun+Dec: after 23 Dec 2011).
  T3  Close pairs (early vs late within 5/10/20 m): change in index and in depth-matched
      qc1Ncs for liquefiable layers, by the 2010 and 2011 labels at the pair.
  T4  Index distributions on the shared footprint by 2010 label (difference-in-differences
      with 500 m cell fixed effects).
  T5  Where are the early soundings? (spatial summary)

Usage: python -I timing_discriminators.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(7)
INDICES = ["LSN", "LPI", "LPIish"]
FEB11 = pd.Timestamp("2011-02-22")
JUN11 = pd.Timestamp("2011-06-13")
DEC11 = pd.Timestamp("2011-12-23")


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def block(E, N, size):
    return np.floor(E / size).astype(int) * 100000 + np.floor(N / size).astype(int)


def boot_auc_diff(df, idx, grp_col, a_val, b_val, n_boot=2000):
    blocks = df["block"].unique()
    rows_by = {b: np.where(df["block"].values == b)[0] for b in blocks}
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(blocks, size=len(blocks), replace=True)
        rows = np.concatenate([rows_by[b] for b in pick])
        d = df.iloc[rows]
        A = d[d[grp_col] == a_val]
        B = d[d[grp_col] == b_val]
        x, y = auc(A["y"], A[idx]), auc(B["y"], B[idx])
        if np.isfinite(x) and np.isfinite(y):
            vals.append(x - y)
    v = np.array(vals)
    return float(np.mean(v)), [float(np.quantile(v, .025)), float(np.quantile(v, .975))], int(len(v))


def main(out_dir):
    out = Path(out_dir)
    cases = pd.read_parquet(out / "cases.parquet")
    cases = cases[cases["variant"] == "inv"].copy()
    cases["early"] = (cases["date"] < FEB11).astype(int)
    cases = cases[cases["date"].notna()]
    early_xy = cases[(cases["event"] == "2011") & (cases["early"] == 1)].drop_duplicates("rec_index")[["E", "N"]].values
    tree = cKDTree(early_xy)
    res = {}

    # ---------- T1: 2016 outcome on shared footprint
    d16 = cases[(cases["event"] == "2016") & cases["y"].notna() & (~cases["cpt_after_event"].astype(bool))].copy()
    d16["dist_early"] = tree.query(d16[["E", "N"]].values, k=1)[0]
    d16["block"] = block(d16["E"].values, d16["N"].values, 1000.0)
    t1 = []
    for r in [100, 200, 400, 800]:
        sub = d16[(d16["early"] == 1) | (d16["dist_early"] <= r)]
        row = {"radius_m": r, "n_early": int((sub["early"] == 1).sum()), "pos_early": int(sub.loc[sub["early"] == 1, "y"].sum()),
               "n_late": int((sub["early"] == 0).sum()), "pos_late": int(sub.loc[sub["early"] == 0, "y"].sum())}
        for idx in INDICES:
            row[f"AUC_{idx}_early"] = auc(sub.loc[sub["early"] == 1, "y"], sub.loc[sub["early"] == 1, idx])
            row[f"AUC_{idx}_late"] = auc(sub.loc[sub["early"] == 0, "y"], sub.loc[sub["early"] == 0, idx])
            m, ci, nb = boot_auc_diff(sub, idx, "early", 0, 1, n_boot=1000)
            row[f"dAUC_{idx}_late_minus_early"] = m
            row[f"dAUC_{idx}_ci95"] = ci
        t1.append(row)
    res["T1_2016_shared_footprint"] = t1

    # ---------- T2: dose-response for the 2010 outcome (late soundings by intervening events)
    d10 = cases[(cases["event"] == "2010") & cases["y"].notna()].copy()
    d10["dist_early"] = tree.query(d10[["E", "N"]].values, k=1)[0]
    d10["block"] = block(d10["E"].values, d10["N"].values, 1000.0)
    d10["n_interv"] = np.select(
        [d10["date"] < FEB11, d10["date"] < JUN11, d10["date"] < DEC11],
        [0, 1, 2], default=3)
    t2 = {}
    for r in [200, 400, 800, 1e9]:
        sub = d10[d10["dist_early"] <= r] if r < 1e9 else d10
        rows = []
        for k, g in sub.groupby("n_interv"):
            row = {"n_intervening_events": int(k), "n": int(len(g)), "base_rate": float(g["y"].mean()),
                   "median_date": str(g["date"].median().date())}
            for idx in INDICES:
                row[f"AUC_{idx}"] = auc(g["y"], g[idx])
            rows.append(row)
        t2[f"radius_{int(r) if r < 1e9 else 'all'}"] = rows
    res["T2_2010_dose_response"] = t2

    # same dose-response for the Feb 2011 outcome (intervening = after the event itself)
    d11 = cases[(cases["event"] == "2011") & cases["y"].notna()].copy()
    d11["dist_early"] = tree.query(d11[["E", "N"]].values, k=1)[0]
    d11["n_interv"] = np.select([d11["date"] < FEB11, d11["date"] < JUN11, d11["date"] < DEC11], [0, 1, 2], default=3)
    t2b = {}
    for r in [200, 400, 800, 1e9]:
        sub = d11[d11["dist_early"] <= r] if r < 1e9 else d11
        rows = []
        for k, g in sub.groupby("n_interv"):
            row = {"n_events_since_or_incl": int(k), "n": int(len(g)), "base_rate": float(g["y"].mean())}
            for idx in INDICES:
                row[f"AUC_{idx}"] = auc(g["y"], g[idx])
            rows.append(row)
        t2b[f"radius_{int(r) if r < 1e9 else 'all'}"] = rows
    res["T2b_2011_by_cpt_timing"] = t2b

    # ---------- T3: close pairs early vs late
    e = cases[(cases["event"] == "2010") & (cases["early"] == 1)].drop_duplicates("rec_index")
    l = cases[(cases["event"] == "2010") & (cases["early"] == 0)].drop_duplicates("rec_index")
    tl = cKDTree(l[["E", "N"]].values)
    pairs = []
    for _, row in e.iterrows():
        idxs = tl.query_ball_point([row["E"], row["N"]], r=20.0)
        for j in idxs:
            lr = l.iloc[j]
            pairs.append({"early_rec": int(row["rec_index"]), "late_rec": int(lr["rec_index"]),
                          "dist_m": float(np.hypot(row["E"] - lr["E"], row["N"] - lr["N"])),
                          "early_date": str(row["date"].date()), "late_date": str(lr["date"].date())})
    pairs = pd.DataFrame(pairs)
    res["T3_n_pairs_le20m"] = int(len(pairs))
    if len(pairs):
        wide = cases.pivot_table(index="rec_index", columns="event", values=["LSN", "LPI", "LPIish", "y", "label"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        P = pairs.merge(wide.add_prefix("E_"), left_on="early_rec", right_index=True).merge(
            wide.add_prefix("L_"), left_on="late_rec", right_index=True)
        P.to_csv(out / "timing_close_pairs.csv", index=False)
        t3 = []
        for rmax in [5, 10, 20]:
            S = P[P["dist_m"] <= rmax]
            row = {"max_dist_m": rmax, "n_pairs": int(len(S)), "n_early_sites": int(S["early_rec"].nunique())}
            for idx in INDICES:
                for ev in ["2010", "2011"]:
                    dlt = np.log1p(S[f"L_{idx}_{ev}"]) - np.log1p(S[f"E_{idx}_{ev}"])
                    row[f"median_dlog1p_{idx}_{ev}"] = float(np.median(dlt)) if len(dlt) else np.nan
                # stratify by label at the early site for 2011 event
                for lab in [0, 1]:
                    s2 = S[S["E_y_2011"] == lab]
                    dl = np.log1p(s2[f"L_{idx}_2011"]) - np.log1p(s2[f"E_{idx}_2011"])
                    row[f"median_dlog1p_{idx}_2011_given_y2011_{lab}"] = float(np.median(dl)) if len(dl) else np.nan
                    row[f"n_given_y2011_{lab}"] = int(len(s2))
                    s3 = S[S["E_y_2010"] == lab]
                    dl3 = np.log1p(s3[f"L_{idx}_2010"]) - np.log1p(s3[f"E_{idx}_2010"])
                    row[f"median_dlog1p_{idx}_2010_given_y2010_{lab}"] = float(np.median(dl3)) if len(dl3) else np.nan
                    row[f"n_given_y2010_{lab}"] = int(len(s3))
                # label agreement between early and late soundings of the pair
            row["label2010_agree_frac"] = float(np.mean(S["E_y_2010"] == S["L_y_2010"])) if len(S) else np.nan
            row["label2011_agree_frac"] = float(np.mean(S["E_y_2011"] == S["L_y_2011"])) if len(S) else np.nan
            t3.append(row)
        res["T3_close_pairs"] = t3

    # ---------- T4: DiD on shared footprint (400 m) with 500 m cell fixed effects
    sub = d10[(d10["dist_early"] <= 400)].copy()
    sub["cell"] = block(sub["E"].values, sub["N"].values, 500.0).astype(str)
    t4 = {}
    for idx in INDICES:
        sub[f"l_{idx}"] = np.log1p(sub[idx])
        m = smf.ols(f"l_{idx} ~ early * y + C(cell)", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["block"]})
        t4[idx] = {k: [float(m.params[k]), float(m.bse[k]), float(m.pvalues[k])] for k in ["early", "y", "early:y"]}
    res["T4_DiD_index_2010_footprint400"] = t4

    # ---------- T5: spatial summary of early soundings
    ee = cases[(cases["event"] == "2011") & (cases["early"] == 1)].drop_duplicates("rec_index")
    la = cases[(cases["event"] == "2011") & (cases["early"] == 0)].drop_duplicates("rec_index")
    res["T5_spatial"] = {
        "early_E_quantiles": [float(q) for q in ee["E"].quantile([.05, .25, .5, .75, .95])],
        "early_N_quantiles": [float(q) for q in ee["N"].quantile([.05, .25, .5, .75, .95])],
        "late_E_quantiles": [float(q) for q in la["E"].quantile([.05, .25, .5, .75, .95])],
        "late_N_quantiles": [float(q) for q in la["N"].quantile([.05, .25, .5, .75, .95])],
        "early_top_name_prefixes": ee["CPTname"].str.extract(r"^([A-Za-z]+)")[0].value_counts().head(15).to_dict(),
        "late_top_name_prefixes": la["CPTname"].str.extract(r"^([A-Za-z]+)")[0].value_counts().head(15).to_dict(),
    }
    (out / "timing_discriminators.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:15000])


if __name__ == "__main__":
    main(sys.argv[1])
