"""Generate LaTeX macro files and tables for both manuscripts from the JSON outputs,
so that no reported number is typed by hand in these places.

Usage: python -I gen_tex_numbers.py <analysis_dir> <manuscripts_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd


def f2(x):
    return f"{x:.2f}"


def ci(v, c):
    return f"{v:.2f} ({c[0]:.2f}--{c[1]:.2f})"


def main(a, m):
    a, m = Path(a), Path(m)
    fn = json.loads((a / "outputs" / "final_numbers.json").read_text())
    sens = json.loads((a / "outputs" / "site_terms_sensitivity.json").read_text())

    # ---------------- Paper A
    A = fn["paperA"]
    lines = [
        r"\newcommand{\TAUISH}{" + ci(A["tau_profile_LPIish_3ev"]["tau"], A["tau_profile_LPIish_3ev"]["tau_ci95"]) + "}",
        r"\newcommand{\ICCISH}{" + ci(A["tau_profile_LPIish_3ev"]["ICC"], A["tau_profile_LPIish_3ev"]["ICC_ci95"]) + "}",
    ]
    (m / "paperA" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    labels = {"inv": "inverse-filtered (primary)", "meas": "measured", "inv_noLS": "inverse, no lateral spreading",
              "meas_noLS": "measured, no lateral spreading"}
    gl = json.loads((a / "outputs" / "site_terms_glmm.json").read_text())
    sd = json.loads((a / "outputs" / "state_dependence.json").read_text())
    for ix in ["LSN", "LPI", "LPIish"]:
        o = gl[ix]["2010+2011+2016"]
        rows.append((labels["inv"], ix, o["tau"], o["ICC"], sd[ix]["g_common"], sd[ix]["g_common_se"]))
    for key in ["meas", "inv_noLS", "meas_noLS"]:
        if key not in sens:
            continue
        for ix in ["LSN", "LPI", "LPIish"]:
            r = sens[key][ix]
            rows.append((labels[key], ix, r["tau"], r["ICC"], r["g_common"], r["g_common_se"]))
    # robustness checks added after the companion analysis of ground change
    rob_f = a / "outputs" / "paperA_robustness.json"
    rob = json.loads(rob_f.read_text()) if rob_f.exists() else {}
    for cfc_key, lab in [("cfc=-0.29", "$C_{FC}=-0.29$"), ("cfc=+0.29", "$C_{FC}=+0.29$")]:
        if cfc_key in rob.get("cfc", {}):
            for ix in ["LSN", "LPI", "LPIish"]:
                r = rob["cfc"][cfc_key][ix]
                rows.append((lab, ix, r["tau"], r["ICC"], r["g_common"], r["g_common_se"]))
    if "threshold_moderate_plus" in rob:
        for ix in ["LSN", "LPI", "LPIish"]:
            r = rob["threshold_moderate_plus"][ix]
            rows.append(("moderate or worse manifestation", ix, r["tau"], r["ICC"], r["g_common"], r["g_common_se"]))
    ep_f = a / "outputs" / "paperA_robustness_epoch.json"
    if ep_f.exists():
        ep = json.loads(ep_f.read_text())
        for grp, lab in [("sounded_before_22Feb2011", "sites sounded before 22 Feb 2011"), ("sounded_2012_2013", "sites sounded in 2012--2013")]:
            for ix in ["LSN", "LPI", "LPIish"]:
                r = ep[grp][ix]
                rows.append((f"{lab} ({ep[grp]['n_sites']:,})", ix, r["tau_3ev"], r["ICC_3ev"], r["g_common"], r["g_common_se"]))
    # all sites: tau from the snapshot taken before the exclusion, gamma from sd_allsites.py (same fit as above)
    snap_f = a / "outputs" / "snapshot_allsites" / "site_terms_glmm.json"
    if snap_f.exists():
        sg = json.loads(snap_f.read_text())
        for ix in ["LSN", "LPI", "LPIish"]:
            o3 = sg[ix]["2010+2011+2016"]
            sa_f = a / "outputs" / f"sd_allsites_{ix}.json"
            sa = json.loads(sa_f.read_text()) if sa_f.exists() else {}
            rows.append(("including ground-improved sites", ix, o3["tau"], o3["ICC"], sa.get("g_common"), sa.get("g_common_se")))
    # forecast table: own site terms versus neighbours only
    if "neighbour_forecast" in rob:
        ft = [r"\begin{table}[tbp]", r"\centering", r"\small",
              r"\caption{AUC gain over the index alone in a held-out earthquake from site terms estimated in the other two earthquakes: the site's own term, and the inverse-distance-weighted mean of other sites' terms between 20~m and $R$ (no co-located history). 95\,\% intervals from 1-km block bootstrap; $w$ is the weight on the site term.}",
              r"\label{tab:neighbour}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Index & Held out & Own term ($w=1$) & Neighbours $\le$100~m & Neighbours $\le$250~m & Sites with neighbours $\le$250~m\\", r"\midrule"]
        for ix in ["LSN", "LPI", "LPIish"]:
            for row in rob["neighbour_forecast"][ix]:
                h = row["held_out"]
                w = "0.25" if h == "2016" else "1.0"
                def g(key):
                    v = row.get(key)
                    return f"${v[0]:+.3f}$ [${v[1][0]:+.3f}$, ${v[1][1]:+.3f}$]" if v else "--"
                ixl = "LPI$_{\\mathrm{ISH}}$" if ix == "LPIish" else ix
                ft.append(f"{ixl} & {h} & {g('gain_own')} & {g(f'gain_nb100_w{w}')} & {g(f'gain_nb250_w{w}')} & {row['share_with_neighbours_250m']:.2f}\\\\")
        ft += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
               r"\par\smallskip\footnotesize For 2016 the neighbour terms are shrunk ($w=0.25$); the own-term column uses $w=1$ in all events.",
               r"\end{table}"]
        (m / "paperA" / "neighbour_table.tex").write_text("\n".join(ft).replace("$-$", "$-$") + "\n", encoding="utf-8")
    t = [r"\begin{table}[tbp]", r"\centering",
         r"\caption{Sensitivity of the site term ($\tau$, ICC; three-event model with epoch-specific 2010 equation) and of the state-dependence coefficient $\gamma$ (joint model of Eq.~\eqref{eq:sd} with normal site terms and equal loadings; standard error in parentheses"
         + ("; -- not estimated" if any(r[4] is None for r in rows) else "")
         + r") to the profile variant, sample, fines-content correlation and outcome definition.}",
         r"\label{tab:sens}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrr}", r"\toprule",
         r"Profile / sample & Index & $\tau$ & ICC & $\gamma$ (SE)\\", r"\midrule"]
    for lab, ix, tau, icc, g, se in rows:
        ixl = "LPI$_{\\mathrm{ISH}}$" if ix == "LPIish" else ix
        gtxt = "--" if g is None else (f"${g:+.2f}$" + (f" ({se:.2f})" if se is not None else ""))
        t.append(f"{lab} & {ixl} & {tau:.2f} & {icc:.2f} & {gtxt}\\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (m / "paperA" / "sensitivity_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------------- Paper B
    B = fn["paperB"]
    T = pd.DataFrame(B["auc_by_event_period"])
    def auc_of(ev, per, ix):
        r = T[(T["event"] == ev) & (T["period"] == per)].iloc[0]
        return f"{r['AUC_' + ix]:.2f} ({r['AUC_' + ix + '_ci95'][0]:.2f}--{r['AUC_' + ix + '_ci95'][1]:.2f})"
    did = B["did_2010_by_feb2011_label"]
    pw = B["pooled_vs_within"]
    lines = [
        r"\newcommand{\AUCPZEROLSN}{" + auc_of("2010", "P0", "LSN") + "}",
        r"\newcommand{\AUCPZEROLPI}{" + auc_of("2010", "P0", "LPI") + "}",
        r"\newcommand{\AUCPZEROISH}{" + auc_of("2010", "P0", "LPIish") + "}",
        r"\newcommand{\AUCPTHREELSN}{" + auc_of("2010", "P3", "LSN") + "}",
        r"\newcommand{\AUCPTHREELPI}{" + auc_of("2010", "P3", "LPI") + "}",
        r"\newcommand{\AUCPTHREEISH}{" + auc_of("2010", "P3", "LPIish") + "}",
        r"\newcommand{\DIDPZEROMONE}{" + ci(did["LSN"]["AUC_P0_m11=1"], did["LSN"]["AUC_P0_m11=1_ci95"]) + "}",
        r"\newcommand{\DIDPTHREEMONE}{" + ci(did["LSN"]["AUC_P3_m11=1"], did["LSN"]["AUC_P3_m11=1_ci95"]) + "}",
        r"\newcommand{\DIDPZEROMZERO}{" + ci(did["LSN"]["AUC_P0_m11=0"], did["LSN"]["AUC_P0_m11=0_ci95"]) + "}",
        r"\newcommand{\DIDPTHREEMZERO}{" + ci(did["LSN"]["AUC_P3_m11=0"], did["LSN"]["AUC_P3_m11=0_ci95"]) + "}",
        r"\newcommand{\DIDLSN}{" + ci(did["LSN"]["DiD"], did["LSN"]["DiD_ci95"]) + "}",
        r"\newcommand{\POOLEDLSN}{" + f2(pw["LSN"]["pooled"]) + "}",
        r"\newcommand{\WITHINLSN}{" + f2(pw["LSN"]["within_weighted"]) + "}",
    ]
    (m / "paperB" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    t = [r"\begin{table}[tbp]", r"\centering", r"\small",
         r"\caption{AUC of the unrefitted indices by earthquake and investigation period, with 95\,\% spatial-block bootstrap intervals. Cells with fewer than ten manifested cases are omitted.}",
         r"\label{tab:period}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrlll}", r"\toprule",
         r"Event & Period & $n$ & Manifested & LSN & LPI & LPI$_{\mathrm{ISH}}$\\", r"\midrule"]
    names = {"P0": "P0 (Sep 2010--Feb 2011)", "P1": "P1 (Feb--Jun 2011)", "P2": "P2 (Jun--Dec 2011)", "P3": "P3 (2012--2013)", "P4": "P4 (2015+)"}
    for _, r in T.iterrows():
        if r["n_pos"] < 10 or (r["n"] - r["n_pos"]) < 10:
            continue
        cells = [f"{r['AUC_' + ix]:.2f} ({r['AUC_' + ix + '_ci95'][0]:.2f}--{r['AUC_' + ix + '_ci95'][1]:.2f})" for ix in ["LSN", "LPI", "LPIish"]]
        t.append(f"{r['event']} & {names[r['period']]} & {int(r['n']):,} & {int(r['n_pos']):,} & " + " & ".join(cells) + r"\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (m / "paperB" / "period_table.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    print("written macro and table files")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
