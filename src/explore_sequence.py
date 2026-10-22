"""First-pass exploration of sequence structure in the Canterbury case table.

1. Reproduction check against the P1/R36 frozen pipeline (LPI, inverse profile).
2. Per-event discrimination of LPI, LSN and LPIish.
3. Cross-event label transitions (raw and index-conditional).
4. CPT timing: soundings tested before versus after the 22 February 2011 event.

Usage: python -I explore_sequence.py <outputs_dir> <p1_l1_parquet>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def auc(y, s):
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def main(out_dir, p1_parquet):
    out = Path(out_dir)
    cases = pd.read_parquet(out / "cases.parquet")
    res = {}

    # 1. reproduction check vs P1 L1 (inverse LPI)
    p1 = pd.read_parquet(p1_parquet)
    p1 = p1[p1["profile"] == "inverse"][["profile_index", "event_code", "LPI"]]
    p1["event"] = p1["event_code"].str[2:]
    mine = cases[cases["variant"] == "inv"][["rec_index", "event", "LPI"]]
    mg = mine.merge(p1, left_on=["rec_index", "event"], right_on=["profile_index", "event"], suffixes=("", "_p1"))
    res["repro_n"] = int(len(mg))
    res["repro_spearman_LPI"] = float(mg[["LPI", "LPI_p1"]].corr(method="spearman").iloc[0, 1])
    res["repro_median_abs_diff_LPI"] = float(np.median(np.abs(mg["LPI"] - mg["LPI_p1"])))

    # 2. per-event AUCs
    lab = cases[cases["y"].notna()].copy()
    tab = []
    for (var, ev), g in lab.groupby(["variant", "event"]):
        row = {"variant": var, "event": ev, "n": len(g), "base_rate": float(g["y"].mean())}
        for idx in ["LPI", "LSN", "LPIish", "thick_trig_10m"]:
            row[f"AUC_{idx}"] = auc(g["y"].values, g[idx].values)
        row["AUC_negcritFS"] = auc(g["y"].values, -np.minimum(g["crit_FS"].values, 5.0))
        tab.append(row)
    res["per_event_auc"] = tab

    # 3. transitions (inverse variant; labels are variant-independent)
    w = cases[cases["variant"] == "inv"].pivot_table(index="rec_index", columns="event", values="y")
    w.columns = [f"y{c}" for c in w.columns]
    sev = cases[cases["variant"] == "inv"].pivot_table(index="rec_index", columns="event", values="label")
    sev.columns = [f"sev{c}" for c in sev.columns]
    idx = cases[cases["variant"] == "inv"].pivot_table(index="rec_index", columns="event", values=["LSN", "LPI", "LPIish", "PGA", "GWT"])
    idx.columns = [f"{a}_{b}" for a, b in idx.columns]
    wide = w.join(sev).join(idx)
    wide.to_parquet(out / "wide_inv.parquet")
    trans = {}
    for a, b in [("y2010", "y2011"), ("y2011", "y2016"), ("y2010", "y2016")]:
        ct = pd.crosstab(wide[a], wide[b])
        trans[f"{a}->{b}"] = ct.to_dict()
        sub = wide[[a, b]].dropna()
        p_b_given_a1 = float(sub.loc[sub[a] == 1, b].mean())
        p_b_given_a0 = float(sub.loc[sub[a] == 0, b].mean())
        trans[f"P({b}=1|{a}=1)"] = p_b_given_a1
        trans[f"P({b}=1|{a}=0)"] = p_b_given_a0
    res["transitions"] = trans

    # conditional on the event index: logit(y_b) ~ log1p(LSN_b) + y_a
    cond = []
    for a, b, evb in [("y2010", "y2011", "2011"), ("y2011", "y2016", "2016"), ("y2010", "y2016", "2016")]:
        for ix in ["LSN", "LPI", "LPIish"]:
            d = wide[[a, b, f"{ix}_{evb}"]].dropna()
            X = sm.add_constant(np.column_stack([np.log1p(d[f"{ix}_{evb}"].values), d[a].values]))
            m = sm.Logit(d[b].values, X).fit(disp=0)
            cond.append({"prior": a, "target": b, "index": ix, "n": len(d),
                         "beta_index": float(m.params[1]), "beta_prior": float(m.params[2]),
                         "OR_prior": float(np.exp(m.params[2])),
                         "OR_prior_lo": float(np.exp(m.conf_int()[2][0])), "OR_prior_hi": float(np.exp(m.conf_int()[2][1])),
                         "AUC_index_only": auc(d[b].values, d[f"{ix}_{evb}"].values),
                         "AUC_index_plus_prior": auc(d[b].values, m.predict(X))})
    # 2016 given both priors
    for ix in ["LSN", "LPI", "LPIish"]:
        d = wide[["y2010", "y2011", "y2016", f"{ix}_2016"]].dropna()
        X = sm.add_constant(np.column_stack([np.log1p(d[f"{ix}_2016"].values), d["y2010"].values, d["y2011"].values]))
        m = sm.Logit(d["y2016"].values, X).fit(disp=0)
        ci = m.conf_int()
        cond.append({"prior": "y2010+y2011", "target": "y2016", "index": ix, "n": len(d),
                     "OR_y2010": float(np.exp(m.params[2])), "OR_y2010_ci": [float(np.exp(ci[2][0])), float(np.exp(ci[2][1]))],
                     "OR_y2011": float(np.exp(m.params[3])), "OR_y2011_ci": [float(np.exp(ci[3][0])), float(np.exp(ci[3][1]))],
                     "AUC_index_only": auc(d["y2016"].values, d[f"{ix}_2016"].values),
                     "AUC_index_plus_priors": auc(d["y2016"].values, m.predict(X))})
    res["conditional_prior"] = cond

    # 4. CPT timing
    s = cases[(cases["variant"] == "inv")].copy()
    s11 = s[(s["event"] == "2011") & s["y"].notna()].copy()
    s11["post"] = s11["cpt_after_event"].astype(int)
    timing = {"n_pre_2011": int((s11["post"] == 0).sum()), "n_post_2011": int((s11["post"] == 1).sum()),
              "base_rate_pre": float(s11.loc[s11["post"] == 0, "y"].mean()),
              "base_rate_post": float(s11.loc[s11["post"] == 1, "y"].mean())}
    for ix in ["LSN", "LPI", "LPIish"]:
        for grp, g in s11.groupby("post"):
            timing[f"AUC_{ix}_post{grp}"] = auc(g["y"].values, g[ix].values)
        X = sm.add_constant(np.column_stack([np.log1p(s11[ix].values), s11["post"].values,
                                             np.log1p(s11[ix].values) * s11["post"].values]))
        m = sm.Logit(s11["y"].values, X).fit(disp=0)
        timing[f"logit_{ix}"] = {"params": [float(v) for v in m.params], "pvalues": [float(v) for v in m.pvalues]}
        timing[f"median_{ix}_pre"] = float(s11.loc[s11["post"] == 0, ix].median())
        timing[f"median_{ix}_post"] = float(s11.loc[s11["post"] == 1, ix].median())
    s10 = s[(s["event"] == "2010") & s["y"].notna()].copy()
    s10["after2011"] = (s10["date"] > pd.Timestamp("2011-02-22")).astype(int)
    timing["n_2010_cases_cpt_before_feb2011"] = int((s10["after2011"] == 0).sum())
    timing["n_2010_cases_cpt_after_feb2011"] = int((s10["after2011"] == 1).sum())
    for ix in ["LSN", "LPI", "LPIish"]:
        for grp, g in s10.groupby("after2011"):
            timing[f"AUC2010_{ix}_after2011_{grp}"] = auc(g["y"].values, g[ix].values)
    s16 = s[(s["event"] == "2016") & s["y"].notna()]
    timing["n_2016_cases_cpt_after_event"] = int(s16["cpt_after_event"].sum())
    res["timing"] = timing

    # date histogram by quarter
    dq = s[s["event"] == "2011"].drop_duplicates("rec_index")["date"].dt.to_period("Q").value_counts().sort_index()
    res["date_quarters"] = {str(k): int(v) for k, v in dq.items()}

    (out / "explore_sequence.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
