"""Supplementary tables for Paper B, generated from the analysis outputs.

Writes manuscripts/paperB/supplement_tables.tex (input by supplement.tex) and
analysis/outputs/raw_unit_check.csv (agreement of parsed NZGD AGS profiles with PRJ-2937).

Usage: python -I gen_si_paperB.py <mat> <analysis_dir> <manuscripts_dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402

MINUS = re.compile(r"(?<![\w$-])-(?=\d)")


def tex(s):
    return MINUS.sub("$-$", str(s))


def table(caption, label, header, rows, align):
    t = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r"\\", r"\midrule"]
    t += [" & ".join(tex(c) for c in r) + r"\\" for r in rows]
    t += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(t)


def unit_check(mat_path, o):
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    meta = pd.read_parquet(o / "nzgd_raw_meta.parquet").drop_duplicates("nzgd_id")
    prof = pd.read_parquet(o / "nzgd_raw_profiles.parquet")
    link = pd.read_parquet(o / "nzgd_link.parquet")
    both = meta.merge(link[["rec_index", "nzgd_id"]], on="nzgd_id")
    rows = []
    for _, r in both.iterrows():
        rec = recs[int(r["rec_index"])]
        dp, qp, fp = (np.atleast_1d(np.asarray(getattr(rec, k), float)) for k in ("depth", "qc", "fs"))
        p = prof[prof["nzgd_id"] == r["nzgd_id"]].sort_values("depth").dropna(subset=["depth", "qc_kpa", "fs_kpa"])
        grid = np.arange(1.0, min(np.nanmax(dp), p["depth"].max()) - 0.1, 0.05)
        if grid.size < 20:
            continue
        qi, fi = np.interp(grid, p["depth"], p["qc_kpa"]), np.interp(grid, p["depth"], p["fs_kpa"])
        qpi, fpi = np.interp(grid, dp, qp), np.interp(grid, dp, fp)
        ok = (qpi > 500) & (qi > 500)
        okf = ok & (fpi > 1)
        rows.append({"loca_id": r["loca_id"], "nzgd_id": int(r["nzgd_id"]), "rec_index": int(r["rec_index"]),
                     "qc_ratio": float(np.median(qi[ok] / qpi[ok])), "fs_ratio": float(np.median(fi[okf] / fpi[okf])),
                     "fs_unit_inferred": r["fs_unit"], "fs_ratio_raw": float(r["fs_ratio_median"])})
    U = pd.DataFrame(rows)
    U.to_csv(o / "raw_unit_check.csv", index=False)
    return U


def main(mat_path, a, m):
    a, m = Path(a), Path(m)
    o = a / "outputs"
    parts = []
    # S1 candidate definitions
    C = pd.read_csv(o / "yang_reproduce_candidates.csv").sort_values("rmse_vs_yang").head(12)
    kind = {"crit_q": "minimum-FS point", "layer_q": "contiguous FS$<$1 layer", "all_q": "all FS$<$1 points"}
    rows = [[r["proc"], "measured" if r["var"] == "meas" else "inverse-filtered", r["demand"].replace("D", ""), kind[r["kind"]],
             f"{r['rmse_vs_yang']:.1f}", f"{r['G1_median']:.1f} ({r['G1_q25']:.1f}--{r['G1_q75']:.1f})",
             f"{r['G2_mean']:.1f}", f"{r['G3_mean']:.1f}", f"{r['G4_mean']:.1f}"] for _, r in C.iterrows()]
    rows.append(["\\multicolumn{5}{l}{Yang et al. (2026), reported}", "", "", "", "", "58 (45--70)", "66", "72", "76"])
    rows[-1] = ["\\multicolumn{5}{l}{Yang et al. (2026), reported}", "58 (45--70)", "$\\approx$66", "$\\approx$72", "$\\approx$76"]
    parts.append(table("Candidate definitions of the liquefiable-layer resistance, ranked by root-mean-square difference from the period statistics reported by Yang et al. (2026). Sample: pre-drilling known and at most 2.5~m, known 2011 manifestation. IB08: equations of Idriss and Boulanger (2008) with fines content from Robertson and Wride (1998); BI14: Boulanger and Idriss (2014). Demand: 2010 or 2011 for all soundings, or the most recent earthquake before the test date (prev).",
                       "tab:s1", ["Procedure", "Profile", "Demand", "Statistic", "RMSE", "G1 median (IQR)", "G2 mean", "G3 mean", "G4 mean"],
                       rows, "llllrrrrr"))
    # S2 strata contrasts
    bp = json.loads((o / "between_period_robustness.json").read_text())
    lab = {"lnqc_sandW_1_5": "saturated sand-like, 1--5 m", "lnqc_sandW_5_10": "saturated sand-like, 5--10 m",
           "lnqc_sandW_1_10": "saturated sand-like, 1--10 m", "lnqc_sand_12_20": "sand-like, 12--20 m",
           "lnqc_clay_1_10": "clay-like, 1--10 m", "lnqc_dry": "unsaturated crust", "lnRf_sandW_1_10": "$\\ln R_f$, saturated sand-like",
           "ic_sandW_1_10": "$I_c$, saturated sand-like"}
    for con, cap in [("G34_vs_P0", "soundings pushed in 2012--2014 and between September 2010 and February 2011"),
                     ("P12_vs_P0", "soundings pushed in 2011 after the February earthquake and between September 2010 and February 2011")]:
        rows = []
        for k, l in lab.items():
            rr = bp["cellFE"][con][k]
            cells = [f"{rr[s][0]:+.2f} [{rr[s][1]:+.2f}, {rr[s][2]:+.2f}]" if s in rr else "--" for s in ["500m", "250m", "100m", "50m", "25m"]]
            pr = bp["pairs"][con]["r50"].get(f"{k}_all")
            cells.append(f"{pr[0]:+.2f} [{pr[1]:+.2f}, {pr[2]:+.2f}] ({pr[3]})" if pr else "--")
            rows.append([l] + cells)
        parts.append(table(f"Difference in stratum medians between {cap}, with cell fixed effects of the given size (1-km block-clustered 95\\,\\% intervals) and for nearest-neighbour pairs within 50~m (block-bootstrap 95\\,\\% intervals; number of pairs in parentheses). Natural-log units except $I_c$.",
                           f"tab:s2{con[:3].lower()}", ["Stratum", "500 m", "250 m", "100 m", "50 m", "25 m", "pairs $\\le$ 50 m"], rows, "lrrrrrr"))
    # S3 within-window slopes
    wr = json.loads((o / "window_robustness.json").read_text())
    rows = []
    for win in ["P2", "P3"]:
        for k in ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sand_12_20"]:
            r_ = wr[win][k]
            cells = []
            for s in ["cellFE_500m", "cellFE_250m", "cellFE_100m", "cellFE_50m", "pairs_50m_sameprog_pd", "pairs_20m_sameprog_pd"]:
                v = r_.get(s)
                cells.append(f"{v['slope']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" if v and "slope" in v else "--")
            rows.append([{"P2": "Jun--Dec 2011", "P3": "Dec 2011--Feb 2016"}[win], lab[k]] + cells)
    parts.append(table("Slope of stratum medians (natural-log units) per decade of days since the earthquake that opens the window, with cell fixed effects or as first differences of same-programme nearest-neighbour pairs with pre-drilled depths within 0.5~m. The window of June--December 2011 has too few close pairs. 95\\,\\% intervals.",
                       "tab:s3", ["Window", "Stratum", "500 m", "250 m", "100 m", "50 m", "pairs $\\le$ 50 m", "pairs $\\le$ 20 m"], rows, "llrrrrrr"))
    # S4 repeat members
    M = pd.read_csv(o / "repeat_members.csv", parse_dates=["date"])
    rows = [[r["fam"], r["suffix"], str(r["date"].date()), int(r["nzgd_id"]), f"{r['sep_m']:.2f}", f"{r['z_end']:.1f}",
             f"{r['gwt']:.2f}", f"{r['crit_q_1_10']:.1f}", f"{r['med_q_1_10']:.1f}"] for _, r in M.iterrows()]
    parts.append(table("Repeat soundings retained at the eight locations (within 6~m of the original). Separation is the NZGD plan distance to the original sounding; identical coordinates for several post-event soundings suggest that their positions were recorded as the original's. Water table: PRJ-2937 2011 value of the original sounding. $q_{c1Ncs}$ statistics at 1--10~m on a 0.02-m grid with the 2011 demand.",
                       "tab:s4", ["Location", "Visit", "Date", "NZGD ID", "Sep. (m)", "Depth (m)", "GWT (m)", "Critical $q_{c1Ncs}$", "Median $q_{c1Ncs}$"],
                       rows, "llrrrrrrr"))
    # S5 increments
    I = pd.read_csv(o / "repeat_increments.csv")
    I = I[I["from"].notna()]
    rows = [[r["fam"], f"{r['from']}$\\rightarrow${r['to']}", int(r["days"]), int(r["n_events"]), f"{r['sep_m']:.2f}",
             f"{r['dlogq_1_10']:+.2f}", f"{r['dlogq_1_4']:+.2f}" if pd.notna(r["dlogq_1_4"]) else "--",
             f"{r['dlogq_4_8']:+.2f}" if pd.notna(r["dlogq_4_8"]) else "--", f"{r['dlogq_8_12']:+.2f}" if pd.notna(r["dlogq_8_12"]) else "--"]
            for _, r in I.iterrows()]
    parts.append(table("Change in the median $\\ln q_c$ of saturated sand-like points (sand-like in both soundings) between consecutive visits, by depth band. Events: number of the earthquakes of 22 February, 13 June and 23 December 2011 between the visits.",
                       "tab:s5", ["Location", "Visits", "Days", "Events", "Sep. (m)", "1--10 m", "1--4 m", "4--8 m", "8--12 m"], rows, "llrrrrrrr"))
    # S6 pre-sequence pairs
    Pp = pd.read_csv(o / "preces_pairs.csv")
    rows = [[int(r["pre"]), r["pre_date"], int(r["post"]), r["post_date"], f"{r['sep_m']:.1f}", f"{r['gwt']:.2f}",
             f"{r['dlogq_sat_sand']:+.2f}", int(r["n_sat"]), f"{r['dlogq_all_1_3']:+.2f}"] for _, r in Pp.iterrows()]
    parts.append(table("Soundings pushed before the sequence and within 5~m of later soundings (NZGD IDs; pairs listed by Tonkin \\& Taylor 2015). Change in median $\\ln q_c$, later minus earlier, for saturated sand-like points between the water table and 3~m, and for all points in 1--3~m.",
                       "tab:s6", ["Earlier", "Date", "Later", "Date", "Sep. (m)", "GWT (m)", "$\\Delta\\ln q_c$ sat. sand", "Points", "$\\Delta\\ln q_c$ 1--3 m"],
                       rows, "lrlrrrrrr"))
    # S7 calibration
    cal = json.loads((o / "calibration_tests.json").read_text())["calibration"]
    rows = []
    for ev in ["2011", "2016"]:
        for r in ["r100", "r200", "r400"]:
            d = cal[ev][r]
            for ix in ["LSN", "LPI"]:
                mg = d.get(ix)
                cl = d.get(ix + "_condlogit500")
                if isinstance(mg, dict) and isinstance(cl, dict):
                    rows.append([ev, r[1:] + " m", ix, d["n"], f"{mg['is_P3'][0]:+.2f} [{mg['is_P3'][1]:+.2f}, {mg['is_P3'][2]:+.2f}]",
                                 f"{cl['is_P3'][0]:+.2f} [{cl['is_P3'][1]:+.2f}, {cl['is_P3'][2]:+.2f}]",
                                 f"{d['base_rate_by_period']['P0']:.3f}", f"{d['base_rate_by_period']['P3']:.3f}"])
    parts.append(table("Calibration by period: coefficient of the indicator for 2012--2013 soundings (relative to soundings pushed between September 2010 and February 2011) in logistic models of manifestation on $\\ln(1+\\mathrm{index})$, on the footprint within the given distance of an early sounding (marginal model, 1-km block-clustered intervals) and within 500-m cells (conditional logit). Base rates are the observed manifestation rates.",
                       "tab:s7", ["Outcome", "Footprint", "Index", "$n$", "Marginal", "Within 500-m cells", "Rate, early", "Rate, 2012--2013"], rows, "lllrrrrr"))
    # S8 unit check
    U = unit_check(mat_path, o)
    rows = [[r["loca_id"], int(r["nzgd_id"]), f"{r['fs_ratio_raw']:.4f}", r["fs_unit_inferred"], f"{r['qc_ratio']:.3f}", f"{r['fs_ratio']:.3f}"]
            for _, r in U.iterrows()]
    parts.append(table("Agreement between parsed NZGD AGS4 profiles and the PRJ-2937 profiles of the same soundings (median ratios on a common 0.05-m grid, points with $q_c>0.5$~MPa). The raw ratio of sleeve friction to cone resistance as stored was used to infer the sleeve-friction unit.",
                       "tab:s8", ["Sounding", "NZGD ID", "Raw $f_s/q_c$", "Inferred $f_s$ unit", "$q_c$ ratio", "$f_s$ ratio"], rows, "lrrlrr"))
    # S10 strata sensitivity to profile variant and water table
    ss = json.loads((o / "strata_sensitivity.json").read_text())
    vlab = {"meas_gwt2011": "measured, 2011 water table (primary)", "inv_gwt2011": "inverse-filtered, 2011 water table",
            "meas_gwt2016": "measured, 2016 water table"}
    rows = []
    for tag, r_ in ss.items():
        for k in ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sand_12_20", "lnqc_dry", "lnRf_sandW_1_10"]:
            v = r_[k]
            rows.append([vlab[tag], lab[k]] + [f"{v[s][0]:+.2f} [{v[s][1]:+.2f}, {v[s][2]:+.2f}]" for s in ["100m", "50m", "pairs50"]])
    parts.append(table("Sensitivity of the difference between soundings pushed in 2012--2014 and before 22 February 2011 to the profile variant and to the water table used to define saturated layers (natural-log units; 95\\,\\% intervals).",
                       "tab:s10", ["Variant", "Stratum", "100-m cells", "50-m cells", "pairs $\\le$ 50 m"], rows, "llrrr"))
    (m / "paperB" / "supplement_tables.tex").write_text("\n".join(parts), encoding="utf-8")
    print("SI tables written;", len(parts), "tables; unit check rows:", len(U))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
