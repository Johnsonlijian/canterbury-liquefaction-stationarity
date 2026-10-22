"""Collect the Paper B numbers on ground change and the reported recovery into one JSON and a LaTeX
macro file (manuscripts/paperB/numbers_recovery.tex), so that the text quotes generated values.

Usage: python -I paperB_numbers.py <analysis_dir> <manuscripts_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_tests import load  # noqa: E402


def j(p):
    return json.loads(Path(p).read_text())


def ci2(v):
    return f"{v[0]:+.2f} ({v[1]:+.2f} to {v[2]:+.2f})"


def ci2u(v):
    return f"{v[0]:.2f} ({v[1]:.2f}--{v[2]:.2f})"


def main(a, m):
    a, m = Path(a), Path(m)
    o = a / "outputs"
    rt, bp, wr = j(o / "recovery_tests.json"), j(o / "between_period_robustness.json"), j(o / "window_robustness.json")
    tv, cal = j(o / "time_vs_event.json"), j(o / "calibration_tests.json")
    pairs = pd.read_csv(o / "repeat_pairs.csv")
    inc = pd.read_csv(o / "repeat_increments.csv")
    win = pd.read_csv(o / "repeat_yang_windows.csv")
    pre = pd.read_csv(o / "preces_pairs.csv")
    # depth of the reproduced critical point, same sample as recovery_tests
    D = load(o)
    Y = pd.read_parquet(o / "yang_metrics.parquet").set_index("rec_index")
    prev = np.where(D["date"] < pd.Timestamp("2011-02-22"), "2010", np.where(D["date"] < pd.Timestamp("2016-02-14"), "2011", "2016"))
    D["crit_z"] = [Y.at[r, f"crit_z_IB08_D{p}_meas"] if r in Y.index else np.nan for r, p in zip(D["rec_index"], prev)]
    zs = D.groupby("G", observed=True)["crit_z"].agg(["mean", "std"]).round(2)
    N = {}
    g = rt["A_group_means"]["Y_rep"]
    for k, nm in zip(["G1", "G2", "G3", "G4"], ["ONE", "TWO", "THREE", "FOUR"]):
        N[f"REPG{nm}MEAN"] = f"{g[k][0]:.1f} ({g[k][1]:.1f}--{g[k][2]:.1f})"
        N[f"REPG{nm}MED"] = f"{rt['A_group_medians']['Y_rep'][k]:.1f}"
        N[f"REPG{nm}N"] = f"{g[k][3]:,}"
        N[f"REPG{nm}Z"] = f"{zs.loc[k, 'mean']:.1f} $\\pm$ {zs.loc[k, 'std']:.1f}"
    N["REPCRITIC"] = f"{D['Y_rep_ic'].median():.2f}"
    grouped = D['Y_rep'].notna() & D['G'].notna()
    n_grouped = int(grouped.sum())
    assert n_grouped == sum(int(g[k][3]) for k in ['G1', 'G2', 'G3', 'G4'])
    N["REPNTOTAL"] = f"{n_grouped:,}"
    N["REPNALLDATES"] = f"{int(D['Y_rep'].notna().sum()):,}"
    N["REPNOUTSIDE"] = f"{int((D['Y_rep'].notna() & D['G'].isna()).sum()):,}"
    sd = pairs[pairs["class"] == "same day"]["dcrit"].abs().sort_values().values
    N["SAMEDAYCRIT"] = f"{sd[0]:.0f} and {sd[-1]:.0f}"
    N["SAMEDAYCRITMAX"] = f"{sd[-1]:.0f}"
    nf = pairs[pairs["class"] == "no event"]["dcrit"]
    N["EVENTFREECRIT"] = f"{nf.min():+.0f} to {nf.max():+.0f}"
    # strata at 100 m cells and other scales
    c100 = bp["cellFE"]["G34_vs_P0"]
    for key, nm in [("lnqc_sandW_1_5", "SATA"), ("lnqc_sandW_5_10", "SATB"), ("lnqc_sand_12_20", "DEEP"), ("lnqc_clay_1_10", "CLAY"),
                    ("lnqc_dry", "DRY"), ("lnRf_sandW_1_10", "RF"), ("ic_sandW_1_10", "IC"), ("lnqc_sandW_1_10", "SATAB")]:
        for sc in ["500m", "250m", "100m", "50m", "25m"]:
            v = c100[key].get(sc)
            if v:
                N[f"STR{nm}{sc.replace('m', 'M').replace('0', 'Z').replace('5', 'F').replace('2', 'T').replace('1', 'O')}"] = ci2(v)
        pr = bp["pairs"]["G34_vs_P0"]["r50"].get(f"{key}_all")
        if pr:
            N[f"STR{nm}PAIRS"] = ci2(pr) + f", $n={pr[3]}$"
        v = bp["cellFE"]["P12_vs_P0"][key].get("100m")
        if v:
            N[f"EARLY{nm}OZZM"] = ci2(v)
        v = bp["cellFE"]["P12_vs_P0"][key].get("500m")
        if v:
            N[f"EARLY{nm}FZZM"] = ci2(v)
    pr = bp["pairs"]["P12_vs_P0"]["r50"].get("lnqc_sandW_1_5_all")
    N["EARLYSATAPAIRS"] = ci2(pr) + f", $n={pr[3]}$"
    N["PAIRSFIFTYN"] = str(bp["pairs"]["G34_vs_P0"]["r50"]["n_pairs"])
    N["PAIRSFIFTYSEP"] = f"{bp['pairs']['G34_vs_P0']['r50']['median_sep_m']:.0f}"
    # predicted change in Ic from the change in Q at constant F (typical sand: Ic = 1.9, Q = 100)
    dlnq = c100["lnqc_sandW_1_10"]["100m"][0]
    N["ICPRED"] = f"{-(3.47 - 2.0) / 1.9 * dlnq / np.log(10):.2f}"
    N["ICOBS"] = f"{c100['ic_sandW_1_10']['100m'][0]:.2f}"
    N["DLNQSATAB"] = f"{dlnq:.2f}"
    # within-window slopes, P3
    p3 = wr["P3"]["lnqc_sandW_1_5"]
    for k, nm in [("cellFE_500m", "FZZ"), ("cellFE_250m", "TFZ"), ("cellFE_100m", "OZZ"), ("cellFE_50m", "FZ"),
                  ("pairs_50m_sameprog_pd", "PFZ"), ("pairs_20m_sameprog_pd", "PTZ")]:
        v = p3[k]
        N[f"WINSATA{nm}"] = f"{v['slope']:+.2f} ({v['lo']:+.2f} to {v['hi']:+.2f})"
        if "n_pairs" in v:
            N[f"WINSATA{nm}N"] = str(v["n_pairs"])
    N["WINDEEPFZ"] = f"{wr['P3']['lnqc_sand_12_20']['cellFE_50m']['slope']:+.2f}"
    N["YANGSLOPE"] = f"{tv['archive']['yang_implied_slope_per_decade']:.2f}"
    # repeat panel
    J = inc[inc["from"].notna()]
    ef = J[J["n_events"] == 0]["dlogq_1_10"]
    N["EFMED"] = f"{ef.median():+.2f}"
    N["EFRANGE"] = f"{ef.drop(ef.idxmax()).min():+.2f} to {ef.drop(ef.idxmax()).max():+.2f}"
    N["EFMAX"] = f"{ef.max():+.2f}"
    toe = J[J["to"] == "e"]
    for c, nm in [("dlogq_1_4", "A"), ("dlogq_4_8", "B"), ("dlogq_8_12", "C")]:
        N[f"TOE{nm}"] = f"{toe[c].median():+.2f}"
        N[f"TOE{nm}RANGE"] = f"{toe[c].min():+.2f} to {toe[c].max():+.2f}"
    feb = J[(J["from"] == "orig") & (J["to"] == "a")]["dlogq_1_10"]
    N["FEBMED"] = f"{feb.median():+.2f}"
    N["FEBRANGE"] = f"{feb.min():+.2f} to {feb.max():+.2f}"
    jun = J[(J["to"] == "d")]["dlogq_1_10"]
    N["JUNMED"] = f"{jun.median():+.2f}"
    N["JUNRANGE"] = f"{jun.min():+.2f} to {jun.max():+.2f}"
    sdq = pairs[pairs["class"] == "same day"]["dlogq_1_10"].abs()
    N["SAMEDAYDLQ"] = f"{sdq.min():.2f} and {sdq.max():.2f}"
    lg = win[(win["fam"] == "BUR-22") & (win["window"] == "liggins")].sort_values("date")
    N["LIGGINSSERIES"] = " $\\rightarrow$ ".join(f"{v:.1f}" for v in lg["mean_q"])
    N["LIGGINSIC"] = " $\\rightarrow$ ".join(f"{v:.2f}" for v in lg["mean_ic"])
    mo = win[(win["fam"] == "AVS-16") & (win["window"] == "morris")].sort_values("date")
    N["MORRISSERIES"] = " $\\rightarrow$ ".join(f"{v:.1f}" for v in mo["mean_q"])
    N["MORRISIC"] = " $\\rightarrow$ ".join(f"{v:.2f}" for v in mo["mean_ic"])
    # raw cone resistance, sleeve friction and friction ratio in the two windows (measured values)
    prof = pd.read_parquet(o / "nzgd_raw_profiles.parquet")
    def window_stats(ids, lo, hi):
        res = []
        for k in ids:
            p = prof[(prof["nzgd_id"] == k) & (prof["depth"] >= lo) & (prof["depth"] <= hi) & (prof["qc_kpa"] > 0)].dropna(subset=["fs_kpa"])
            rf = p["fs_kpa"].clip(lower=0.1) / p["qc_kpa"] * 100
            res.append((p["qc_kpa"].median() / 1000, p["fs_kpa"].median(), rf.median()))
        return res
    lg_ids = [284, 55509, 56287, 55504, 55460]
    lw = window_stats(lg_ids, 1.6, 7.22)
    N["LIGGINSQC"] = " $\\rightarrow$ ".join(f"{v[0]:.1f}" for v in lw)
    N["LIGGINSRFORIG"] = f"{lw[0][2]:.2f}"
    N["LIGGINSRFLATER"] = f"{min(v[2] for v in lw[1:]):.2f}--{max(v[2] for v in lw[1:]):.2f}"
    mw = window_stats([110, 55507, 56289], 3.09, 4.04)
    N["MORRISQC"] = " $\\rightarrow$ ".join(f"{v[0]:.1f}" for v in mw)
    N["MORRISFS"] = " $\\rightarrow$ ".join(f"{v[1]:.0f}" for v in mw)
    N["MORRISRF"] = " $\\rightarrow$ ".join(f"{v[2]:.1f}" for v in mw)
    post = pre[pre["post_date"] >= "2012-01-01"]["dlogq_sat_sand"].sort_values()
    N["PRECES"] = ", ".join(f"{v:+.2f}" for v in post)
    N["PRECESTEN"] = f"{pre[pre['post_date'] < '2012-01-01']['dlogq_sat_sand'].iloc[0]:+.2f}"
    # calibration
    c11 = cal["calibration"]["2011"]
    c16 = cal["calibration"]["2016"]
    N["CALELEVENCELL"] = ci2u(c11["r100"]["LSN_condlogit500"]["is_P3"]).replace("(", "(").replace("--", " to ")
    N["CALSIXTEENCELL"] = ci2u(c16["r100"]["LSN_condlogit500"]["is_P3"]).replace("--", " to ")
    N["CALSIXTEENCELLFOUR"] = ci2u(c16["r400"]["LSN_condlogit500"]["is_P3"]).replace("--", " to ")
    N["CALELEVENMARG"] = ci2u(c11["r100"]["LSN"]["is_P3"]).replace("--", " to ")
    N["CALSIXTEENMARG"] = ci2u(c16["r100"]["LSN"]["is_P3"]).replace("--", " to ")
    (o / "paperB_numbers.json").write_text(json.dumps(N, indent=1))
    import re
    minus = re.compile(r"(?<![\w$-])-(?=\d)")
    lines = ["\\newcommand{\\" + k + "}{" + minus.sub("$-$", v) + "}" for k, v in N.items()]
    (m / "paperB" / "numbers_recovery.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in N.items():
        print(f"{k:<22} {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
