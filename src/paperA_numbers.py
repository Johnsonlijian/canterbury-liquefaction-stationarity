"""Generate every Paper A number quoted in the text as LaTeX macros (paperA/numbers_main.tex) plus
the baseline-AUC and update-factor tables, from the JSON outputs.

Usage: python -I paperA_numbers.py <outputs_dir> <paperA_dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

MINUS = re.compile(r"(?<![\w$-])-(?=\d)")
IX = ["LSN", "LPI", "LPIish"]


def j(p):
    return json.loads(Path(p).read_text())


def rng(vals, fmt="{:.2f}", sep="--"):
    lo, hi = min(vals), max(vals)
    def fz(v):
        s = fmt.format(v)
        return s.lstrip("-+") if float(s) == 0 else s  # drop the sign of a formatted zero only
    a, b = fz(lo), fz(hi)
    if lo < -0.005 and sep == "--":
        sep = " to "
    return a if a == b else f"{a}{sep}{b}"


def ci(v, c, fmt="{:.2f}"):
    sep = " to " if min(c) < 0 else "--"
    return f"{fmt.format(v)} ({fmt.format(c[0])}{sep}{fmt.format(c[1])})"


def main(out_dir, paper_dir):
    o, P = Path(out_dir), Path(paper_dir)
    gl, fn, sd = j(o / "site_terms_glmm.json"), j(o / "final_numbers.json")["paperA"], j(o / "state_dependence.json")
    sens, ex, dg = j(o / "site_terms_sensitivity.json"), j(o / "explore_sequence.json"), j(o / "site_terms_diagnostics.json")
    geo, uf = j(o / "geomorph_site_terms.json"), j(o / "update_factors.json")
    N = {}
    m3 = gl["LSN"]["2010+2011+2016"]
    N["NCASES"] = f"{m3['n_rows']:,}"
    N["NSITES"] = f"{m3['n_sites']:,}"
    N["NSITESALLTHREE"] = f"{sd['LSN']['n_sites']:,}"
    pe = {r["event"]: r for r in ex["per_event_auc"] if r["variant"] == "inv"}
    N["BASETEN"] = f"{100 * pe['2010']['base_rate']:.0f}"
    N["BASEELEVEN"] = f"{100 * pe['2011']['base_rate']:.0f}"
    N["BASESIXTEEN"] = f"{100 * pe['2016']['base_rate']:.1f}"
    # variance decomposition
    taus = [fn[f"tau_profile_{ix}_3ev"]["tau"] for ix in IX]
    iccs = [fn[f"tau_profile_{ix}_3ev"]["ICC"] for ix in IX]
    N["TAURANGE"] = rng(taus)
    N["ICCPCTRANGE"] = rng([100 * v for v in iccs], "{:.0f}")
    for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
        r = fn[f"tau_profile_{ix}_3ev"]
        N[f"TAUCI{nm}"] = ci(r["tau"], r["tau_ci95"])
        N[f"ICCCI{nm}"] = ci(r["ICC"], r["ICC_ci95"])
    N["ICCCILO"] = f"{min(fn[f'tau_profile_{ix}_3ev']['ICC_ci95'][0] for ix in IX):.2f}"
    N["ICCCIHI"] = f"{max(fn[f'tau_profile_{ix}_3ev']['ICC_ci95'][1] for ix in IX):.2f}"
    N["LRMIN"] = f"{int(np.floor(min(gl[ix]['2010+2011+2016']['LR_vs_tau0'] for ix in IX) / 10) * 10)}"
    N["TAUNOTEN"] = rng([fn[f"tau_profile_{ix}_2011_2016"]["tau"] for ix in IX])
    N["TAUNOEPOCH"] = rng([gl[ix]["2010+2011+2016_noEpochSplit"]["tau"] for ix in IX])
    if "meas" in sens:
        N["TAUMEAS"] = rng([sens["meas"][ix]["tau"] for ix in IX])
        N["GMEAS"] = rng([sens["meas"][ix]["g_common"] for ix in IX], "{:+.2f}", " to ")
    if "inv_noLS" in sens:
        N["TAUNOLS"] = rng([sens["inv_noLS"][ix]["tau"] for ix in IX])
        N["GNOLS"] = rng([sens["inv_noLS"][ix]["g_common"] for ix in IX], "{:+.2f}", " and ")
        N["GNOLSSE"] = f"{np.mean([sens['inv_noLS'][ix]['g_common_se'] for ix in IX]):.2f}"
    # recurrence in raw outcomes
    tr = ex["transitions"]
    N["PELEVENGIVENTEN"] = f"{100 * tr['P(y2011=1|y2010=1)']:.1f}"
    N["PELEVENGIVENNOTTEN"] = f"{100 * tr['P(y2011=1|y2010=0)']:.1f}"
    N["PSIXTEENGIVENELEVEN"] = f"{100 * tr['P(y2016=1|y2011=1)']:.1f}"
    N["PSIXTEENGIVENNOTELEVEN"] = f"{100 * tr['P(y2016=1|y2011=0)']:.1f}"
    t = tr["y2011->y2016"]
    # crosstab(index=prior, columns=later).to_dict() is keyed by the later outcome first
    n16 = int(t["1.0"].get("0.0", 0) + t["1.0"].get("1.0", 0))
    N["NSIXTEENMAN"] = f"{n16:,}"
    N["NSIXTEENMANALSOELEVEN"] = f"{int(t['1.0']['1.0']):,}"
    cp = {(r["prior"], r["target"], r["index"]): r for r in ex["conditional_prior"]}
    for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
        r = cp[("y2010", "y2011", ix)]
        N[f"ORNAIVE{nm}"] = f"{r['OR_prior']:.1f} ({r['OR_prior_lo']:.1f}--{r['OR_prior_hi']:.1f})"
    # state dependence
    for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
        r = sd[ix]
        N[f"GCOMMON{nm}"] = f"{r['g_common']:+.2f}"
        N[f"GCOMMONSE{nm}"] = f"{r['g_common_se']:.2f}"
        N[f"ORSD{nm}"] = f"{np.exp(r['g_common']):.2f}"
        N[f"ORSDCI{nm}"] = f"{np.exp(r['g_common'] - 1.96 * r['g_common_se']):.2f}--{np.exp(r['g_common'] + 1.96 * r['g_common_se']):.2f}"
        N[f"PSD{nm}"] = f"{r['p_common']:.2f}"
        N[f"GELEVEN{nm}"] = f"{r['g11']:+.2f} ({r['g11_se']:.2f})"
        N[f"GSIXTEEN{nm}"] = f"{r['g16']:+.1f} ({r['g16_se']:.1f})"
    N["ORSDRANGE"] = rng([np.exp(sd[ix]["g_common"]) for ix in IX])
    N["PSEPMIN"] = f"{min(sd[ix]['p_sep'] for ix in IX):.2f}"
    N["TAUWITHG"] = rng([sd[ix]["tau_common"] for ix in IX])
    N["TAUWITHOUTG"] = rng([sd[ix]["tau_H0"] for ix in IX])
    # spatial and geomorphic structure
    vg = dg["D2_variogram_LSN"]
    bins = vg["bins"]
    N["SVNEAR"] = f"{bins[0]['semivariance']:.2f}"
    far = [b["semivariance"] for b in bins if b["lo_m"] >= 1000]
    N["SILL"] = f"{np.mean(far):.2f}" if far else "--"
    N["NNCORRFIFTY"] = f"{vg['nn_corr_le_50m'][0]:.2f}"
    lf = geo["site_term_by_landform"]
    def lfv(name):
        k = [kk for kk in lf if kk.strip() == name][0]
        return lf[k]
    N["GEOESTUARY"] = f"{lfv('Holocene estuary plain')['mean']:+.2f}"
    N["GEOESTUARYN"] = f"{int(lfv('Holocene estuary plain')['size']):,}"
    N["GEORIVERPLAIN"] = f"{lfv('Holocene river plain')['mean']:+.2f}"
    N["GEOSWAMP"] = f"{lfv('Holocene river swamp basin')['mean']:+.2f}"
    N["GEODUNE"] = f"{lfv('Holocene dune')['mean']:+.2f}"
    N["GEODUNEN"] = f"{int(lfv('Holocene dune')['size']):,}"
    N["GEORAISED"] = f"{lfv('Holocene river channel raised')['mean']:+.2f}"
    d = geo["log_dist_channel_coef_given_landform"]
    N["DISTCOEF"] = f"{d[0]:+.3f}"
    N["DISTSE"] = f"{d[1]:.3f}"
    N["DISTP"] = "<0.001" if d[2] < 0.001 else f"={d[2]:.3f}"
    N["RTWOLF"] = f"{100 * geo['R2_landform_only']:.1f}"
    N["RTWOCPT"] = f"{100 * geo['R2_landform_plus_cpt']:.1f}"
    N["SPEARPGASIG"] = f"{dg['D3_abs_dhat_vs_PGAsig_spearman']:.2f}"
    # forecast value
    for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
        for row in gl[ix]["LOEO"]:
            h = {"2010": "TEN", "2011": "ELEVEN", "2016": "SIXTEEN"}[row["held_out"]]
            N[f"GAIN{h}{nm}"] = ci(row["AUC_gain"], row["AUC_gain_ci95"], "{:.3f}")
    gains1011 = [r["AUC_gain"] for ix in IX for r in gl[ix]["LOEO"] if r["held_out"] in ("2010", "2011")]
    N["GAINRANGE"] = rng(gains1011)
    N["PRIORONLYAUC"] = rng([r["AUC_prior_labels_only"] for ix in IX for r in gl[ix]["LOEO"]])
    g16 = [r["AUC_gain"] for ix in IX for r in gl[ix]["LOEO"] if r["held_out"] == "2016"]
    N["GAINSIXTEENRANGE"] = f"{max(g16):+.3f} to {min(g16):+.3f}"
    co = [dg["D1_2016_transfer"][ix]["in_event_coef_dhat"] for ix in IX]
    N["COEFDHATRANGE"] = rng([c[0] for c in co])
    pmax = max(c[2] for c in co) if len(co[0]) > 2 else 1.0
    N["COEFDHATP"] = "p<10^{-6}" if pmax < 1e-6 else (f"p<{pmax:.0e}" if pmax < 1e-3 else f"p\\le{pmax:.3f}")
    lam = [0.0, 0.1, 0.25, 0.5, 1.0]
    best = [lam[int(np.argmax([dg["D1_2016_transfer"][ix][f"AUC_oracle_slope_lambda_{l}"] for l in lam]))] for ix in IX]
    N["LAMBDABEST"] = rng(best, "{:g}")
    # tables
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{Discrimination of the three events by each index computed with event-specific demand and water table (inverse-filtered profiles; no refitting).}",
             r"\label{tab:baseline}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Event & Case histories & Base rate & AUC LSN & AUC LPI & AUC LPI$_{\mathrm{ISH}}$\\", r"\midrule"]
    names = {"2010": "2010 Darfield", "2011": "2011 Christchurch", "2016": "2016 Christchurch"}
    for ev in ["2010", "2011", "2016"]:
        r = pe[ev]
        lines.append(f"{names[ev]} & {r['n']:,} & {r['base_rate']:.3f} & {r['AUC_LSN']:.3f} & {r['AUC_LPI']:.3f} & {r['AUC_LPIish']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (P / "baseline_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bh = uf["LSN"]["by_history"]
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{Posterior site terms by outcome history in 2010 and 2011 (LSN model fitted to these two events), implied odds factors for a future event, and the observed 2016 manifestation rate.}",
             r"\label{tab:update}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"History & Sites & Mean site term & Odds factor & Factor ($\lambda=0.5$) & Observed 2016 rate\\", r"\midrule"]
    for h in ["2010 and 2011", "2011 only", "2010 only", "neither"]:
        r = bh[h]
        lines.append(f"{h} & {r['n']:,} & ${r['mean_site_term']:+.2f}$ & {r['odds_factor_lambda1']:.2f} & {r['odds_factor_lambda0.5']:.2f} & "
                     f"{r['observed_2016_rate']:.3f} ($n={r['n_2016']:,}$)\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (P / "update_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    N["UPDBOTHTERM"] = f"{bh['2010 and 2011']['mean_site_term']:+.2f}"
    N["UPDBOTHOF"] = f"{bh['2010 and 2011']['odds_factor_lambda1']:.1f}"
    N["UPDBOTHOFHALF"] = f"{bh['2010 and 2011']['odds_factor_lambda0.5']:.1f}"
    N["UPDNEITHERTERM"] = f"{bh['neither']['mean_site_term']:+.2f}"
    N["UPDNEITHEROF"] = f"{bh['neither']['odds_factor_lambda1']:.2f}"
    N["UPDNEITHEROFHALF"] = f"{bh['neither']['odds_factor_lambda0.5']:.2f}"
    N["UPDRATES"] = ", ".join(f"{100 * bh[h]['observed_2016_rate']:.1f}\\,\\%" for h in ["2010 and 2011", "2011 only", "2010 only", "neither"])
    # robustness of the site term
    if (o / "paperA_robustness.json").exists():
        rob = j(o / "paperA_robustness.json")
        if "cfc" in rob:
            N["CFCTAU"] = rng([rob["cfc"][k][ix]["tau"] for k in rob["cfc"] for ix in IX])
            N["CFCICC"] = rng([100 * rob["cfc"][k][ix]["ICC"] for k in rob["cfc"] for ix in IX], "{:.0f}")
        if "threshold_moderate_plus" in rob:
            th = rob["threshold_moderate_plus"]
            N["THRTAU"] = rng([th[ix]["tau"] for ix in IX])
            N["THRICC"] = rng([100 * th[ix]["ICC"] for ix in IX], "{:.0f}")
            N["THRNPOS"] = ", ".join(f"{th['n_pos'][e]:,}" for e in ["2010", "2011", "2016"])
        if "neighbour_forecast" in rob:
            nb = rob["neighbour_forecast"]
            own = [r["gain_own"][0] for ix in IX for r in nb[ix] if r["held_out"] in ("2010", "2011")]
            n250 = [r["gain_nb250_w1.0"][0] for ix in IX for r in nb[ix] if r["held_out"] in ("2010", "2011")]
            n250h = [r["gain_nb250_w0.5"][0] for ix in IX for r in nb[ix] if r["held_out"] in ("2010", "2011")]
            N["NBGAINRANGE"] = rng(n250 + n250h, "{:.3f}")
            N["NBOWNRANGE"] = rng(own, "{:.3f}")
            N["NBCORR"] = rng([r["corr_dnb_own_250m"] for ix in IX for r in nb[ix]])
            N["NBSHARE"] = f"{100 * min(r['share_with_neighbours_250m'] for ix in IX for r in nb[ix]):.0f}"
            sig = sum(1 for ix in IX for r in nb[ix] if r["held_out"] in ("2010", "2011") for w in ("1.0", "0.5")
                      if r[f"gain_nb250_w{w}"][1][0] > 0)
            N["NBSIGCOUNT"] = f"{sig}"
            N["NBSIGTEXT"] = "in none" if sig == 0 else f"in only {sig}"
    if (o / "paperA_robustness_epoch.json").exists():
        ep = j(o / "paperA_robustness_epoch.json")
        N["EPOCHPZEROTAU"] = rng([ep["sounded_before_22Feb2011"][ix]["tau_3ev"] for ix in IX])
        N["EPOCHPZEROICC"] = rng([100 * ep["sounded_before_22Feb2011"][ix]["ICC_3ev"] for ix in IX], "{:.0f}")
        N["EPOCHPZERON"] = f"{ep['sounded_before_22Feb2011']['n_sites']:,}"
        N["EPOCHPTHREETAU"] = rng([ep["sounded_2012_2013"][ix]["tau_3ev"] for ix in IX])
        N["EPOCHPTHREEICC"] = rng([100 * ep["sounded_2012_2013"][ix]["ICC_3ev"] for ix in IX], "{:.0f}")
        N["EPOCHPTHREEN"] = f"{ep['sounded_2012_2013']['n_sites']:,}"
    snap = o / "snapshot_allsites" / "site_terms_glmm.json"
    if snap.exists():
        sg = j(snap)
        N["ALLSITESTAU"] = rng([sg[ix]["2010+2011+2016"]["tau"] for ix in IX])
        N["ALLSITESN"] = f"{sg['LSN']['2010+2011+2016']['n_sites']:,}"
    # identification of event memory (robust normal model, mixtures, free 2016 loading)
    prim = "excluding_ground_improved"
    if (o / "sd_robust.json").exists():
        rb = j(o / "sd_robust.json")
        # point estimates from state_dependence.json (same fit as the sensitivity table), profile
        # interval from the multi-start profile of sd_robust.py
        for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
            c95 = rb[prim][ix]["g_profile_ci95"]
            N[f"SDG{nm}"] = f"{sd[ix]['g_common']:+.2f}" + (f" ({c95[0]:+.1f} to {c95[1]:+.1f})" if c95 else "")
            N[f"SDOR{nm}"] = f"{np.exp(sd[ix]['g_common']):.1f}"
            N[f"SDP{nm}"] = f"{sd[ix]['p_common']:.3f}"
        N["SDORRANGE"] = rng([np.exp(sd[ix]["g_common"]) for ix in IX], "{:.1f}")
        # all sites (122 ground-improved soundings included): multi-start fit of sd_allsites.py
        g_all = [j(o / f"sd_allsites_{ix}.json")["g_common"] for ix in IX if (o / f"sd_allsites_{ix}.json").exists()]
        if len(g_all) < len(IX) and "all_sites" in rb:
            g_all = [rb["all_sites"][ix]["g_common"] for ix in IX if ix in rb["all_sites"]]
        if g_all:
            N["SDGALLSITES"] = rng(g_all, "{:+.2f}", " to ")
        if (o / "sd_allsites_LSN.json").exists():
            # ground-improved sites that have all three outcomes and so enter the three-event model
            N["SDGINSITES"] = f"{j(o / 'sd_allsites_LSN.json')['n_sites'] - rb[prim]['LSN']['n_sites']:,}"
        if (o / "paperA_gi_diagnostics.json").exists():
            # placebo: drop 122 randomly chosen sites from the full sample instead of the ground-improved ones
            rd = j(o / "paperA_gi_diagnostics.json").get("random_drop_122_LSN_g_common", [])
            if rd:
                N["GIRANDDROP"] = rng(rd, "{:+.2f}", " to ")
                N["GIRANDN"] = f"{len(rd)}"
        N["SDGELEVEN"] = rng([rb[prim][ix]["g11"] for ix in IX], "{:+.2f}", " to ")
        N["SDTAUH"] = rng([rb[prim][ix]["tau_H0"] for ix in IX])
        N["SDLAMH"] = rng([rb[prim][ix]["lam_H0"] for ix in IX])
    if (o / "sd_mixture.json").exists():
        mx = j(o / "sd_mixture.json")
        mp = mx[prim]
        gs = [mp[ix][f"K{k}"]["g"] for ix in mp for k in (2, 3, 4) if mp[ix][f"K{k}"]["LR"] >= 0]
        N["MIXGRANGE"] = rng(gs, "{:+.2f}", " to ")
        N["MIXORRANGE"] = rng([np.exp(g) for g in gs], "{:.1f}")
        gains = []
        for ix in mp:
            aic_n = 2 * mp[ix]["normal"]["nll"] + 2 * 11
            aic_m = min(mp[ix][f"K{k}"]["AIC_g"] for k in (2, 3, 4) if mp[ix][f"K{k}"]["LR"] >= 0)
            gains.append(aic_n - aic_m)
        N["MIXAICGAIN"] = rng(gains, "{:.0f}")
    if (o / "sd_loading.json").exists():
        ld = j(o / "sd_loading.json")
        fr = ld[prim]["LSN"]["free_lam16"]
        fx = ld[prim]["LSN"]["lam16_fixed_1"]
        bestfr = min(fr, key=lambda p: p["nll"])
        N["FREELAMBDA"] = f"{bestfr['lam16']:.2f}"
        N["FREEGBEST"] = f"{bestfr['g']:+.1f}"
        N["FREEDEVZERO"] = f"{[p['dev'] for p in fr if abs(p['g']) < 1e-9][0]:.1f}"
        N["FIXEDDEVZERO"] = f"{[p['dev'] for p in fx if abs(p['g']) < 1e-9][0]:.1f}"
        N["FREEVSFIXED"] = f"{2 * (min(p['nll'] for p in fx) - bestfr['nll']):.1f}"
    # repeatability by pair of earthquakes, loading test, sequential forecast of 2016
    if (o / "pairwise_tau.json").exists():
        pw = j(o / "pairwise_tau.json")
        for pr, nm in [("2010-2011", "TENELEVEN"), ("2011-2016", "ELEVENSIXTEEN"), ("2010-2016", "TENSIXTEEN")]:
            vals = [pw["primary"][ix][pr]["tau"] for ix in IX if ix in pw["primary"]]
            N[f"PAIR{nm}"] = rng(vals)
            lrs = [max(pw["primary"][ix][pr]["LR_vs_tau0"], 0.0) for ix in IX if ix in pw["primary"]]
            N[f"PAIR{nm}LR"] = rng(lrs, "{:.1f}")
            ps = [pw["primary"][ix][pr]["p"] for ix in IX if ix in pw["primary"]]
            N[f"PAIR{nm}PMAX"] = f"{max(ps):.2g}" if max(ps) >= 1e-4 else "<10^{-4}"
            his = [pw["primary"][ix][pr]["tau_ci95_grid"][1] for ix in IX if ix in pw["primary"] and pw["primary"][ix][pr]["tau_ci95_grid"]]
            if his:
                N[f"PAIR{nm}UPPER"] = f"{max(his):.2f}"
            for s, snm in [("all_sites", "ALL"), ("p0_sites", "PZERO")]:
                r = pw.get(s, {}).get("LSN", {}).get(pr)
                if r:
                    N[f"PAIR{nm}{snm}"] = f"{r['tau']:.2f}"
                    N[f"PAIR{nm}{snm}LR"] = f"{r['LR_vs_tau0']:.1f}"
        if "sequential_2016" in pw:
            sq = pw["sequential_2016"]
            for tag, nm in [("from_2011", "ELEVEN"), ("from_2010", "TEN"), ("from_both", "BOTH")]:
                gains = [sq[ix][f"{tag}_w0.5"]["gain"] for ix in IX if ix in sq]
                N[f"SEQ{nm}"] = rng(gains, "{:+.3f}", " to ")
                sig = sum(1 for ix in IX if ix in sq for w in ("1.0", "0.5", "0.25") if sq[ix][f"{tag}_w{w}"]["ci"][0] > 0)
                N[f"SEQ{nm}SIG"] = f"{sig}"
            N["SEQAUCINDEX"] = rng([sq[ix]["AUC_index"] for ix in IX if ix in sq])
    if (o / "sd_loading.json").exists():
        ld = j(o / "sd_loading.json")
        lrs, lams = [], []
        for ix in ld["excluding_ground_improved"]:
            fx = min(ld["excluding_ground_improved"][ix]["lam16_fixed_1"], key=lambda p: p["nll"])
            fr = min(ld["excluding_ground_improved"][ix]["free_lam16"], key=lambda p: p["nll"])
            lrs.append(2 * (fx["nll"] - fr["nll"]))
            lams.append(fr["lam16"])
        N["LOADLR"] = rng(lrs, "{:.0f}")
        for ix, nm in [("LSN", "LSN"), ("LPIish", "ISHX")]:
            if ix in ld["excluding_ground_improved"]:
                fr = ld["excluding_ground_improved"][ix]["free_lam16"]
                b = min(fr, key=lambda p: p["nll"])
                ins = [p["g"] for p in fr if p["dev"] <= 3.84]
                N[f"FREEG{nm}"] = f"{b['g']:+.1f}"
                N[f"FREECI{nm}"] = f"{min(ins):+.1f} to {max(ins):+.1f}"
                N[f"FREEGRIDMAX"] = f"{max(p['g'] for p in fr):+.1f}"
        N["LOADLAMBDA"] = rng([abs(v) for v in lams], "{:.2f}")
        N["LOADLAMBDAMAX"] = f"{max(abs(v) for v in lams):.2f}"
    lines = ["\\newcommand{\\" + k + "}{" + MINUS.sub("$-$", v) + "}" for k, v in N.items()]
    # any upper-case macro used in main.tex but not generated (nor defined in numbers.tex) gets a
    # visible placeholder, so a missing number can never pass silently
    used = set(re.findall(r"\\([A-Z]{4,})\b", (P / "main.tex").read_text(encoding="utf-8")))
    other = set(re.findall(r"\\newcommand\{\\([A-Z]+)\}", (P / "numbers.tex").read_text(encoding="utf-8"))) if (P / "numbers.tex").exists() else set()
    missing = sorted(used - set(N) - other)
    lines += ["\\newcommand{\\" + k + "}{\\textbf{??}}" for k in missing]
    if missing:
        print("MISSING MACROS:", ", ".join(missing))
    (P / "numbers_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (o / "paperA_numbers.json").write_text(json.dumps(N, indent=1))
    for k, v in N.items():
        print(f"{k:<26} {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
