"""Publication figures for Paper A (site memory) and Paper B (case-history provenance).

All numbers are read from the JSON/parquet outputs of the analysis scripts; nothing is
typed in by hand. Palette: validated categorical slots (blue, orange, aqua; all-pairs CVD
pass) for events, an ordinal single-hue ramp for investigation periods, and a blue-red
diverging map with a grey midpoint for site terms.

Usage: python -I make_figures.py <analysis_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

EVENT_COL = {"2010": "#2a78d6", "2011": "#eb6834", "2016": "#1baf7a"}
EVENT_LAB = {"2010": "2010 Darfield $M_w$7.1", "2011": "2011 Christchurch $M_w$6.2", "2016": "2016 Christchurch $M_w$5.7"}
PERIOD_COL = {"P0": "#104281", "P1": "#256abf", "P2": "#3987e5", "P3": "#86b6ef"}
PERIOD_LAB = {"P0": "Sep 2010–Feb 2011", "P1": "Feb–Jun 2011", "P2": "Jun–Dec 2011", "P3": "2012–2013"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a12b2b"])
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False,
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK)


def load(a, name):
    return json.loads((a / "outputs" / name).read_text())


def gm3_outlines(a):
    from pyproj import Transformer
    from shapely.geometry import shape
    with open(a / "external" / "gm3_units_chch.geojson", encoding="utf-8") as fh:
        gj = json.load(fh)
    tr = Transformer.from_crs(4326, 2193, always_xy=True)
    lines = []
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        geom = shape(g)
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            x, y = p.exterior.xy
            X, Y = tr.transform(np.array(x), np.array(y))
            lines.append((X, Y))
    return lines


# ======================================================================= Paper A
def paper_a(a, outdir):
    snd = pd.read_parquet(a / "outputs" / "soundings.parquet")
    geo = pd.read_parquet(a / "outputs" / "soundings_geo.parquet")
    wide = pd.read_parquet(a / "outputs" / "wide_inv.parquet")
    S = snd.merge(geo, on="rec_index").merge(wide[["y2010", "y2011", "y2016"]], left_on="rec_index", right_index=True, how="left")
    outlines = gm3_outlines(a)
    xmin, xmax = S["X2193"].quantile(0.001) - 1500, S["X2193"].quantile(0.999) + 1500
    ymin, ymax = S["Y2193"].quantile(0.001) - 1500, S["Y2193"].quantile(0.999) + 1500

    # ---- A1: site map with sequence classes
    def seq_class(r):
        if pd.isna(r["y2010"]) or pd.isna(r["y2011"]):
            return "incomplete labels"
        if r["y2010"] == 1 and r["y2011"] == 1:
            return "2010 and 2011"
        if r["y2011"] == 1:
            return "2011 only"
        if r["y2010"] == 1:
            return "2010 only"
        return "neither 2010 nor 2011"
    S["seq"] = S.apply(seq_class, axis=1)
    seq_style = {"neither 2010 nor 2011": ("#c3c2b7", 3), "2011 only": ("#eb6834", 3), "2010 only": ("#1baf7a", 6),
                 "2010 and 2011": ("#2a78d6", 3), "incomplete labels": ("#ffffff", 0)}
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 85 * MM), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for X, Y in outlines:
        ax.plot(X, Y, color="#d9d8d1", lw=0.25, zorder=1)
    for k in ["neither 2010 nor 2011", "2011 only", "2010 only", "2010 and 2011"]:
        g = S[S["seq"] == k]
        ax.scatter(g["X2193"], g["Y2193"], s=seq_style[k][1], c=seq_style[k][0], lw=0, label=f"{k} (n={len(g):,})", zorder=3)
    s16 = S[S["y2016"] == 1]
    ax.scatter(s16["X2193"], s16["Y2193"], s=12, facecolors="none", edgecolors=INK, lw=0.5,
               label=f"also manifested in 2016 (n={len(s16)})", zorder=4)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", fontsize=6, markerscale=2.2, handletextpad=0.3, borderaxespad=0.2)
    ax.plot([xmax - 6000, xmax - 1000], [ymax - 1500] * 2, color=INK, lw=1.2)
    ax.text(xmax - 3500, ymax - 1100, "5 km", ha="center", fontsize=6)
    ax.set_title("Manifestation history at the CPT sites", loc="left")
    panel(ax, "a")
    # b: transition probabilities raw vs index-conditional
    ax = axs[1]
    ex = load(a, "explore_sequence.json")
    tr = ex["transitions"]
    cats = [("P(y2011=1|y2010=1)", "P(y2011=1|y2010=0)", "2011 | 2010"),
            ("P(y2016=1|y2011=1)", "P(y2016=1|y2011=0)", "2016 | 2011"),
            ("P(y2016=1|y2010=1)", "P(y2016=1|y2010=0)", "2016 | 2010")]
    xs = np.arange(len(cats))
    p1 = [tr[c[0]] for c in cats]
    p0 = [tr[c[1]] for c in cats]
    ax.bar(xs - 0.18, p1, width=0.34, color="#2a78d6", label="prior manifestation")
    ax.bar(xs + 0.18, p0, width=0.34, color="#c3c2b7", label="no prior manifestation")
    for x, v in zip(xs - 0.18, p1):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
    for x, v in zip(xs + 0.18, p0):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([c[2] for c in cats])
    ax.set_ylabel("Observed manifestation probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Recurrence in the raw outcomes", loc="left")
    panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "A1_sites_and_recurrence")

    # ---- A2b: repeatable variance by pair of earthquakes (non-stationarity)
    idxs = ["LSN", "LPI", "LPIish"]
    pw_f = a / "outputs" / "pairwise_tau.json"
    if pw_f.exists():
        pw = json.loads(pw_f.read_text())
        fig, axs = plt.subplots(1, 2, figsize=(170 * MM, 60 * MM), sharey=True)
        pairs = ["2010-2011", "2011-2016", "2010-2016"]
        plab = ["2010 and 2011", "2011 and 2016", "2010 and 2016"]
        ax = axs[0]
        for i, ix in enumerate(idxs):
            for k, pr in enumerate(pairs):
                r = pw["primary"].get(ix, {}).get(pr)
                if not r:
                    continue
                x = k + (i - 1) * 0.18
                ci_ = r.get("tau_ci95_grid")
                if ci_ and ci_[1] > ci_[0]:
                    ax.plot([x, x], ci_, color=["#2a78d6", "#eb6834", "#1baf7a"][i], lw=1.1)
                ax.plot(x, r["tau"], "o", ms=4.5, color=["#2a78d6", "#eb6834", "#1baf7a"][i], label=ix if k == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(plab)
        ax.set_ylabel(r"Site-to-site s.d. $\tau$ (logit units)")
        ax.set_ylim(-0.1, 3.5)
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Repeatability by pair of earthquakes", loc="left"); panel(ax, "a")
        ax = axs[1]
        samp = [("primary", "ground-improved sites excluded", "#2a78d6"), ("all_sites", "all sites", "#898781"),
                ("p0_sites", "sounded before 22 Feb 2011", "#104281")]
        for i, (s, lab, col) in enumerate(samp):
            for k, pr in enumerate(pairs):
                r = pw.get(s, {}).get("LSN", {}).get(pr)
                if not r:
                    continue
                x = k + (i - 1) * 0.18
                ci_ = r.get("tau_ci95_grid")
                if ci_ and ci_[1] > ci_[0]:
                    ax.plot([x, x], ci_, color=col, lw=1.1)
                ax.plot(x, r["tau"], "o", ms=4.5, color=col, label=lab if k == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(plab)
        ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), fontsize=6)
        ax.set_title("LSN by sample", loc="left"); panel(ax, "b")
        fig.tight_layout()
        save(fig, outdir / "A2b_pairwise_repeatability")

    # ---- A2: tau / ICC with profile CIs
    fn = load(a, "final_numbers.json")["paperA"]
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 55 * MM))
    specs = [("2011_2016", "2011+2016"), ("3ev", "2010+2011+2016\n(2010 split by epoch)")]
    idxs = ["LSN", "LPI", "LPIish"]
    for j, (key, lab) in enumerate(specs):
        for i, ix in enumerate(idxs):
            r = fn[f"tau_profile_{ix}_{key}"]
            y = j * 4 + i
            axs[0].errorbar(r["tau"], y, xerr=[[r["tau"] - r["tau_ci95"][0]], [r["tau_ci95"][1] - r["tau"]]],
                            fmt="o", ms=4, color=EVENT_COL["2010"] if i == 0 else ("#eb6834" if i == 1 else "#1baf7a"),
                            elinewidth=1.2, capsize=0)
            axs[1].errorbar(r["ICC"], y, xerr=[[r["ICC"] - r["ICC_ci95"][0]], [r["ICC_ci95"][1] - r["ICC"]]],
                            fmt="o", ms=4, color=EVENT_COL["2010"] if i == 0 else ("#eb6834" if i == 1 else "#1baf7a"),
                            elinewidth=1.2, capsize=0)
    yt = [0, 1, 2, 4, 5, 6]
    ytl = [f"{ix}" for ix in idxs] * 2
    for ax in axs:
        ax.set_yticks(yt); ax.set_yticklabels(ytl)
        ax.set_ylim(6.6, -1.5)
        ax.text(0.01, -0.95, "events 2011 + 2016", transform=ax.get_yaxis_transform(), fontsize=6, color=INK2)
        ax.text(0.01, 3.1, "events 2010 + 2011 + 2016 (2010 split by test epoch)", transform=ax.get_yaxis_transform(),
                fontsize=6, color=INK2)
    axs[0].set_xlabel(r"Site-to-site s.d. $\tau$ (logit units)")
    axs[1].set_xlabel(r"Repeatable share of latent residual variance (ICC)")
    axs[1].set_yticklabels([])
    axs[0].set_title("Repeatable site term", loc="left"); panel(axs[0], "a")
    axs[1].set_title("Intraclass correlation", loc="left"); panel(axs[1], "b")
    fig.tight_layout()
    save(fig, outdir / "A2_site_term_variance")

    # ---- A3: event memory under different assumptions about the site term
    cond = ex["conditional_prior"]
    out_dir = a / "outputs"
    rb = json.loads((out_dir / "sd_robust.json").read_text()) if (out_dir / "sd_robust.json").exists() else None
    mx = json.loads((out_dir / "sd_mixture.json").read_text()) if (out_dir / "sd_mixture.json").exists() else None
    ld = json.loads((out_dir / "sd_loading.json").read_text()) if (out_dir / "sd_loading.json").exists() else None
    prim = "excluding_ground_improved"
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 80 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    cols3 = {"LSN": "#2a78d6", "LPI": "#eb6834", "LPIish": "#1baf7a"}
    groups = ["naive, 2011 given 2010", "normal site terms", "two-point site terms", "three-point site terms", "free 2016 loading"]
    for gi_, grp in enumerate(groups):
        for i, ix in enumerate(idxs):
            y = gi_ + (i - 1) * 0.22
            lo = hi = None
            if grp.startswith("naive"):
                r = [c for c in cond if c["prior"] == "y2010" and c["target"] == "y2011" and c["index"] == ix][0]
                o, lo, hi = r["OR_prior"], r["OR_prior_lo"], r["OR_prior_hi"]
            elif grp == "normal site terms" and rb:
                r = rb[prim][ix]
                o = np.exp(r["g_common"])
                if r["g_profile_ci95"]:
                    lo, hi = np.exp(r["g_profile_ci95"][0]), np.exp(r["g_profile_ci95"][1])
            elif grp.endswith("point site terms") and mx:
                k = "K2" if grp.startswith("two") else "K3"
                r = mx[prim][ix][k]
                if r["LR"] < 0:
                    continue
                o = np.exp(r["g"])
            elif grp == "free 2016 loading" and ld and ix in ld[prim]:
                pr = ld[prim][ix]["free_lam16"]
                best = min(pr, key=lambda p: p["nll"])
                o = np.exp(best["g"])
                inside = [p["g"] for p in pr if p["dev"] <= 3.84]
                lo, hi = np.exp(min(inside)), np.exp(max(inside))
            else:
                continue
            if lo is not None:
                ax.errorbar(o, y, xerr=[[o - lo], [hi - o]], fmt="o", ms=4, color=cols3[ix], elinewidth=1.1, capsize=0,
                            label=ix if gi_ == 0 else None)
            else:
                ax.plot(o, y, "o", ms=4, mfc="white", mec=cols3[ix], mew=1.1)
    ax.axvline(1.0, color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups); ax.invert_yaxis()
    ax.set_xlabel("Odds ratio for prior manifestation\n(95% profile interval; open: point estimate only)")
    if ld and "LSN" in ld[prim]:
        prl = ld[prim]["LSN"]["free_lam16"]
        bl = min(prl, key=lambda p: p["nll"])
        if abs(bl["g"] - max(p["g"] for p in prl)) < 1e-9:
            ax.annotate("LSN: at edge of\nexplored range", xy=(np.exp(bl["g"]), len(groups) - 1 - 0.22), xytext=(2.6, len(groups) - 1.62),
                        fontsize=5.8, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("Event memory depends on the assumed site term", loc="left"); panel(ax, "a")
    ax = axs[1]
    if ld:
        for ix, ls in [("LSN", "-"), ("LPIish", "--")]:
            if ix not in ld[prim]:
                continue
            for key, col, lab in [("lam16_fixed_1", "#2a78d6", "2016 loading fixed at 1"), ("free_lam16", "#eb6834", "2016 loading free")]:
                pr = ld[prim][ix][key]
                ax.plot([p["g"] for p in pr], [p["dev"] for p in pr], ls, marker="o", ms=3, lw=1.1, color=col,
                        label=f"{lab}, {ix}" if True else None)
        ax.axhline(3.84, color=INK2, lw=0.6)
        ax.text(-1.45, 4.3, "95% threshold", fontsize=5.8, color=INK2)
        ax.set_ylim(0, 30)
    ax.set_xlabel(r"State-dependence coefficient $\gamma$")
    ax.set_ylabel("Profile deviance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.8)
    ax.set_title("A free 2016 loading flattens the profile", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "A3_state_dependence")

    # ---- A4: site-term map + semivariogram
    eb = pd.read_parquet(a / "outputs" / "site_terms_eb_LSN.parquet").merge(geo, on="rec_index")
    diag = load(a, "site_terms_diagnostics.json")
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 82 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    for X, Y in outlines:
        ax.plot(X, Y, color="#e6e5df", lw=0.25, zorder=1)
    lim = float(np.quantile(np.abs(eb["d_mean"]), 0.98))
    sc = ax.scatter(eb["X2193"], eb["Y2193"], c=eb["d_mean"], s=3, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), lw=0, zorder=3)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Empirical-Bayes site term (logit)\n< 0: less manifestation than index predicts", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.plot([xmax - 6000, xmax - 1000], [ymin + 1200] * 2, color=INK, lw=1.2)
    ax.text(xmax - 3500, ymin + 1700, "5 km", ha="center", fontsize=6)
    ax.set_title("Site terms from three events (LSN model)", loc="left"); panel(ax, "a")
    ax = axs[1]
    vg = pd.DataFrame(diag["D2_variogram_LSN"]["bins"])
    mid = 0.5 * (vg["lo_m"] + vg["hi_m"])
    ax.plot(mid, vg["semivariance"], "o-", color="#2a78d6", ms=4, lw=1.2)
    ax.axhline(diag["D2_variogram_LSN"]["variance_dhat"], color=INK2, lw=0.6)
    ax.text(mid.iloc[-1], diag["D2_variogram_LSN"]["variance_dhat"] * 1.02, "variance of site terms", ha="right", va="bottom", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Separation distance (m)")
    ax.set_ylabel("Empirical semivariance of site terms")
    ax.set_ylim(0, None)
    ax.set_title("Spatial structure", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "A4_site_term_map_variogram")

    # ---- A5: site terms vs geomorphology and channel distance
    gm = load(a, "geomorph_site_terms.json")
    D = eb.copy()
    lf_counts = D["gm3_landform"].value_counts()
    keep = [k for k in lf_counts.index if lf_counts[k] >= 40]
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 66 * MM), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    order = D[D["gm3_landform"].isin(keep)].groupby("gm3_landform")["d_mean"].mean().sort_values().index
    for k, lf in enumerate(order):
        v = D.loc[D["gm3_landform"] == lf, "d_mean"].values
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(len(v))
        q1, q3 = np.quantile(v, [0.25, 0.75])
        ax.plot([q1, q3], [k, k], color="#b7d3f6", lw=4, solid_capstyle="butt")
        ax.errorbar(m, k, xerr=1.96 * se, fmt="o", ms=4, color="#2a78d6", elinewidth=1.2, capsize=0)
        ax.text(1.01, k, f"n={len(v):,}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=5.8, color=MUTED)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([s.strip() for s in order])
    ax.set_xlabel("Site term (mean ± 95% CI; bar = interquartile range)")
    ax.set_xlim(-1.3, 1.7)
    ax.set_title("By mapped GM3 landform", loc="left"); panel(ax, "a")
    ax = axs[1]
    bins = [0, 50, 100, 200, 400, 800, 1600]
    D["db"] = pd.cut(D["dist_channel_m"], bins + [1e7])
    g = D.groupby("db", observed=True)["d_mean"]
    means, ses, ns = g.mean(), g.std() / np.sqrt(g.size()), g.size()
    xs = np.arange(len(means))
    ax.errorbar(xs, means.values, yerr=1.96 * ses.values, fmt="o-", ms=4, color="#2a78d6", elinewidth=1.2, lw=1.0, capsize=0)
    ax.axhline(0, color=INK2, lw=0.6)
    labs = ["0–50", "50–100", "100–200", "200–400", "400–800", "800–1600", ">1600"][:len(xs)]
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30)
    for x, n in zip(xs, ns.values):
        ax.text(x, ax.get_ylim()[0] if False else -0.95, f"{n}", ha="center", fontsize=5.6, color=MUTED)
    ax.set_ylim(-1.1, 0.5)
    ax.set_xlabel("Distance to mapped channel unit (m)")
    ax.set_ylabel("Site term (mean ± 95% CI)")
    coef = gm["log_dist_channel_coef_given_landform"]
    ax.text(0.98, 0.95, f"slope on ln(1+d) | landform = {coef[0]:.3f} (SE {coef[1]:.3f})", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=INK2)
    ax.set_title("By distance to channel (free-face proxy)", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "A5_site_terms_geomorphology")

    # ---- A6: LOEO forecast value + 2016 shrinkage
    gl = load(a, "site_terms_glmm.json")
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 58 * MM))
    ax = axs[0]
    for i, ix in enumerate(idxs):
        for j, row in enumerate(gl[ix]["LOEO"]):
            y = j * 4 + i
            g, ci = row["AUC_gain"], row["AUC_gain_ci95"]
            ax.errorbar(g, y, xerr=[[g - ci[0]], [ci[1] - g]], fmt="o", ms=4,
                        color=["#2a78d6", "#eb6834", "#1baf7a"][i], elinewidth=1.2, capsize=0,
                        label=ix if j == 0 else None)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks([1, 5, 9]); ax.set_yticklabels(["held-out 2010", "held-out 2011", "held-out 2016"]); ax.invert_yaxis()
    ax.set_xlabel("AUC gain from other-event site terms (95% CI)")
    ax.legend(loc="center left", fontsize=6)
    ax.set_title("Forecast value of observed performance", loc="left"); panel(ax, "a")
    ax = axs[1]
    lam = [0.0, 0.1, 0.25, 0.5, 1.0]
    for i, ix in enumerate(idxs):
        r = diag["D1_2016_transfer"][ix]
        v = [r[f"AUC_oracle_slope_lambda_{l}"] for l in lam]
        ax.plot(lam, v, "o-", ms=3.5, lw=1.2, color=["#2a78d6", "#eb6834", "#1baf7a"][i], label=ix)
    ax.set_xlabel(r"Weight $\lambda$ on 2010–2011 site terms")
    ax.set_ylabel("2016 AUC (in-event slope; diagnostic)")
    ax.legend(loc="lower center", fontsize=6, ncol=3, bbox_to_anchor=(0.5, 0.0))
    ax.set_title("2010-2011 site terms do not help 2016", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "A6_forecast_value")


# ======================================================================= Paper B
def paper_b(a, outdir):
    snd = pd.read_parquet(a / "outputs" / "soundings.parquet")
    geo = pd.read_parquet(a / "outputs" / "soundings_geo.parquet")
    S = snd.merge(geo, on="rec_index")
    fn = load(a, "final_numbers.json")["paperB"]
    tm = load(a, "timing_matched.json")
    tl = load(a, "timing_label_checks.json")
    tp = load(a, "timing_protocol_check.json")

    # ---- B1: timeline + map of early vs 2012-13 soundings
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    q = S["date"].dt.to_period("Q").value_counts().sort_index()
    qs = [p.start_time for p in q.index]
    cols = []
    for t in qs:
        if t < pd.Timestamp("2011-02-22"):
            cols.append(PERIOD_COL["P0"])
        elif t < pd.Timestamp("2011-06-13"):
            cols.append(PERIOD_COL["P1"])
        elif t < pd.Timestamp("2011-12-23"):
            cols.append(PERIOD_COL["P2"])
        elif t < pd.Timestamp("2014-01-01"):
            cols.append(PERIOD_COL["P3"])
        else:
            cols.append("#c3c2b7")
    ax.bar(qs, q.values, width=80, color=cols, align="edge")
    ev = {"2010-09-04": "Darfield", "2011-02-22": "Christchurch", "2011-06-13": "June 2011", "2011-12-23": "Dec 2011", "2016-02-14": "Feb 2016"}
    for d, lab in ev.items():
        t = pd.Timestamp(d)
        ax.axvline(t, color=INK, lw=0.7)
        ax.text(t, ax.get_ylim()[1] * 0.97 if False else 1750, " " + lab, rotation=90, va="top", ha="right", fontsize=6, color=INK)
    ax.set_ylim(0, 1800)
    ax.set_ylabel("CPT soundings per quarter")
    ax.set_xlim(pd.Timestamp("2010-07-01"), pd.Timestamp("2017-04-01"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PERIOD_COL[p], label=PERIOD_LAB[p]) for p in ["P0", "P1", "P2", "P3"]],
              loc="upper right", bbox_to_anchor=(0.80, 1.0), fontsize=6, title="test period", title_fontsize=6)
    ax.set_title("When the 5,668 PRJ-2937 soundings were pushed", loc="left"); panel(ax, "a")
    ax = axs[1]
    outlines = gm3_outlines(a)
    xmin, xmax = S["X2193"].quantile(0.001) - 1500, S["X2193"].quantile(0.999) + 1500
    ymin, ymax = S["Y2193"].quantile(0.001) - 1500, S["Y2193"].quantile(0.999) + 1500
    for X, Y in outlines:
        ax.plot(X, Y, color="#e6e5df", lw=0.25, zorder=1)
    late = S[(S["date"] >= pd.Timestamp("2011-12-23")) & (S["date"] < pd.Timestamp("2014-01-01"))]
    early = S[S["date"] < pd.Timestamp("2011-02-22")]
    ax.scatter(late["X2193"], late["Y2193"], s=3, c=PERIOD_COL["P3"], lw=0, label=f"2012–2013 (n={len(late):,})", zorder=2)
    ax.scatter(early["X2193"], early["Y2193"], s=3, c=PERIOD_COL["P0"], lw=0, label=f"Sep 2010–Feb 2011 (n={len(early)})", zorder=3)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", markerscale=3, fontsize=6)
    ax.set_title("Where they were pushed", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "B1_timeline_map")

    # ---- B2: AUC by event x period (LSN, LPI, LPIish small multiples)
    T = pd.DataFrame(fn["auc_by_event_period"])
    T = T[T["period"].isin(["P0", "P1", "P2", "P3"])]
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 62 * MM), sharey=True)
    for k, ix in enumerate(["LSN", "LPI", "LPIish"]):
        ax = axs[k]
        for j, ev in enumerate(["2010", "2011", "2016"]):
            for i, per in enumerate(["P0", "P1", "P2", "P3"]):
                r = T[(T["event"] == ev) & (T["period"] == per)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if r["n_pos"] < 10 or (r["n"] - r["n_pos"]) < 10:
                    continue  # too few manifested (or non-manifested) cases for a stable AUC
                x = j * 5 + i
                v, ci = r[f"AUC_{ix}"], r[f"AUC_{ix}_ci95"]
                ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=4, color=PERIOD_COL[per],
                            elinewidth=1.2, capsize=0, label=PERIOD_LAB[per] if (j == 0 and k == 0) else None)
        ax.axhline(0.5, color=INK2, lw=0.6)
        ax.set_xticks([1.5, 6.5, 11.5]); ax.set_xticklabels(["2010\nDarfield", "2011\nChristchurch", "2016\nChristchurch"])
        ax.set_title(f"{ix}", loc="left")
        panel(ax, "abc"[k])
    axs[0].set_ylabel("AUC of the unrefitted index (95% CI)")
    axs[0].set_ylim(0.35, 1.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.5, title="sounding test period", title_fontsize=6.5,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, outdir / "B2_auc_by_event_period")

    # ---- B3: footprint-matched difference vs radius
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 56 * MM), sharey=True)
    for k, ev in enumerate(["2010", "2011"]):
        ax = axs[k]
        fp = tm[f"{ev}_inv"]["footprint"]
        for i, ix in enumerate(["LSN", "LPI", "LPIish"]):
            r = np.array([f["radius_m"] for f in fp])
            d = np.array([f[f"dAUC_{ix}_early_minus_late"] for f in fp])
            lo = np.array([f[f"dAUC_{ix}_ci95"][0] for f in fp])
            hi = np.array([f[f"dAUC_{ix}_ci95"][1] for f in fp])
            off = (i - 1) * 0.06
            ax.errorbar(r * np.exp(off), d, yerr=[d - lo, hi - d], fmt="o-", ms=3.5, lw=1.0, elinewidth=1.0,
                        color=["#2a78d6", "#eb6834", "#1baf7a"][i], capsize=0, label=ix)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 400]); ax.set_xticklabels(["50", "100", "200", "400"])
        ax.set_xlabel("Late soundings kept within r of an early sounding (m)")
        ax.set_title(f"{EVENT_LAB[ev]} outcome", loc="left"); panel(ax, "ab"[k])
    axs[0].set_ylabel("AUC(early) − AUC(late), same footprint")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, outdir / "B3_footprint_matched")

    # ---- B4: DiD by Feb 2011 label
    did = fn["did_2010_by_feb2011_label"]
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 60 * MM), sharey=True)
    cols3 = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, (m, title) in enumerate([(1, "Sites that manifested in February 2011"), (0, "Sites that did not")]):
        ax = axs[k]
        for i, ix in enumerate(["LSN", "LPI", "LPIish"]):
            r = did[ix]
            pts = []
            for j, per in enumerate(["P0", "P3"]):
                v = r[f"AUC_{per}_m11={m}"]
                c = r[f"AUC_{per}_m11={m}_ci95"]
                x = j + (i - 1) * 0.1
                ax.errorbar(x, v, yerr=[[v - c[0]], [c[1] - v]], fmt="o", ms=4, color=cols3[i], elinewidth=1.1, capsize=0,
                            label=ix if (k == 0 and j == 0) else None)
                pts.append((x, v))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cols3[i], lw=0.8)
            if k == 0:
                pass
        ax.axhline(0.5, color=INK2, lw=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["tested\nSep 2010–Feb 2011", "tested\n2012–2013"])
        ax.set_xlim(-0.5, 1.5)
        n0, n3 = did["LSN"][f"n_P0_m11={m}"], did["LSN"][f"n_P3_m11={m}"]
        ax.text(0.5, 0.03, f"n = {n0} and {n3:,}", transform=ax.transAxes, ha="center", fontsize=6, color=MUTED)
        ax.set_title(title, loc="left"); panel(ax, "ab"[k])
    axs[0].set_ylabel("2010 Darfield AUC (95% CI)")
    axs[0].set_ylim(0.35, 0.95)
    r = did["LSN"]
    axs[1].text(0.98, 0.97, f"difference-in-differences (LSN)\n{r['DiD']:.2f} [{r['DiD_ci95'][0]:.2f}, {r['DiD_ci95'][1]:.2f}]",
                transform=axs[1].transAxes, ha="right", va="top", fontsize=6, color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, outdir / "B4_did_feb2011")

    # ---- B5: mechanism checks
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 58 * MM))
    ax = axs[0]
    tr = pd.DataFrame(tp["truncation_test_2010"])
    tr["per"] = tr["period"].str[:2]
    tr = tr[(tr["footprint"] == "all") & tr["per"].isin(["P0", "P1", "P2", "P3"])]
    for per in ["P0", "P1", "P2", "P3"]:
        g = tr[tr["per"] == per].sort_values("z0")
        ax.plot(g["z0"], g["AUC_LSN"], "o-", ms=3.5, lw=1.0, color=PERIOD_COL[per], label=PERIOD_LAB[per])
    ax.axhline(0.5, color=INK2, lw=0.6)
    ax.set_xlabel("Data above this depth discarded (m)")
    ax.set_ylabel("2010 AUC (LSN)")
    ax.set_ylim(0.34, 0.85)
    ax.legend(fontsize=5.6, loc="lower center", ncol=2, columnspacing=0.8, handlelength=1.5)
    ax.set_title("Not pre-drilling or shallow truncation", loc="left"); panel(ax, "a")
    ax = axs[1]
    l3 = pd.DataFrame(tl["L3_neighbour_label_agreement"])
    l3 = l3[(l3["radius_m"] == 50) & l3["period"].isin(["P0", "P1", "P2", "P3"])]
    xs = np.arange(len(l3))
    ax.bar(xs - 0.18, l3["agree2010_exact"], width=0.34, color="#2a78d6", label="2010 severity class")
    ax.bar(xs + 0.18, l3["agree2011_binary"], width=0.34, color="#eb6834", label="2011 manifested yes/no")
    ax.set_xticks(xs); ax.set_xticklabels([PERIOD_LAB[p] for p in l3["period"]], rotation=20, fontsize=6)
    ax.set_ylabel("Agreement with nearest neighbour ≤ 50 m")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Label resolution differs by programme", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "B5_mechanism_checks")

    # ---- B6: reproduction of the reported trend and negative-control strata
    rt = load(a, "recovery_tests.json")
    bp = load(a, "between_period_robustness.json")
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    groups = ["G1", "G2", "G3", "G4"]
    glab = ["Sep 2010–\nFeb 2011", "Feb–Dec\n2011", "2012", "2013–\n2014"]
    mean = [rt["A_group_means"]["Y_rep"][g] for g in groups]
    xs = np.arange(4)
    ax.errorbar(xs - 0.1, [m[0] for m in mean], yerr=[[m[0] - m[1] for m in mean], [m[2] - m[0] for m in mean]], fmt="o",
                ms=4.5, color="#2a78d6", elinewidth=1.2, capsize=0, label="this study, mean (95% block CI)")
    ax.plot(xs - 0.1, [rt["A_group_medians"]["Y_rep"][g] for g in groups], "o", ms=4.5, mfc="white", mec="#2a78d6", mew=1.0,
            label="this study, median")
    yang = [(0, 58.0, "median"), (1, 66.0, "mean"), (2, 72.0, "mean"), (3, 76.0, "mean")]
    ax.plot([y[0] + 0.1 for y in yang], [y[1] for y in yang], "D", ms=4.5, color="#eb6834",
            label="Yang et al. (2026): median (IQR), then means")
    ax.plot([0.1, 0.1], [45, 70], color="#eb6834", lw=1.2)
    for x, m in zip(xs, mean):
        ax.text(x - 0.1, 41.0, f"n={m[3]:,}", ha="center", fontsize=5.8, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels(glab)
    ax.set_ylim(38, 92)
    ax.set_ylabel("$q_{c1Ncs}$ at the minimum-FS point")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("The reported trend is reproduced", loc="left"); panel(ax, "a")
    ax = axs[1]
    strata = [("lnqc_sandW_1_5", "saturated sand-like, 1–5 m"), ("lnqc_sandW_5_10", "saturated sand-like, 5–10 m"),
              ("lnqc_sand_12_20", "sand-like, 12–20 m"), ("lnqc_clay_1_10", "clay-like ($I_c$ > 2.95), 1–10 m"),
              ("lnqc_dry", "unsaturated crust"), ("lnRf_sandW_1_10", "friction ratio, saturated sand-like")]
    for k, (c, lab) in enumerate(strata):
        for j, (con, col, lab2) in enumerate([("G34_vs_P0", "#2a78d6", "2012–2014 vs Sep 2010–Feb 2011"),
                                              ("P12_vs_P0", "#86b6ef", "2011 (same programme) vs Sep 2010–Feb 2011")]):
            v = bp["cellFE"][con][c].get("100m")
            if not v:
                continue
            ax.errorbar(v[0], k + (j - 0.5) * 0.28, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=4, color=col,
                        elinewidth=1.2, capsize=0, label=lab2 if k == 0 else None)
            if j == 0:
                ax.text(1.01, k, f"n={v[3]:,}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.8, color=MUTED)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(strata))); ax.set_yticklabels([s[1] for s in strata]); ax.invert_yaxis()
    ax.set_xlabel("Change in median ln $q_c$ (or ln $R_f$) within 100 m cells (95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=6)
    ax.set_title("Only the liquefiable zone changed", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "B6_reproduction_strata")

    # ---- B7: between-period change vs elapsed-time slope under tighter location control
    wr = load(a, "window_robustness.json")
    tv = load(a, "time_vs_event.json")
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), sharex=False)
    scales = ["500m", "250m", "100m", "50m", "25m"]
    sc_x = np.arange(len(scales) + 1)
    sc_lab = ["500 m", "250 m", "100 m", "50 m", "25 m", "pairs\n≤ 50 m"]
    st_cols = [("lnqc_sandW_1_5", "#2a78d6", "saturated, 1–5 m"), ("lnqc_sandW_5_10", "#eb6834", "saturated, 5–10 m"),
               ("lnqc_sand_12_20", "#1baf7a", "12–20 m")]
    ax = axs[0]
    for i, (c, col, lab) in enumerate(st_cols):
        pts = []
        for k, s in enumerate(scales):
            v = bp["cellFE"]["G34_vs_P0"][c].get(s)
            if v:
                pts.append((k + (i - 1) * 0.12, v[0], v[1], v[2]))
        pr = bp["pairs"]["G34_vs_P0"]["r50"].get(f"{c}_all")
        if pr:
            pts.append((len(scales) + (i - 1) * 0.12, pr[0], pr[1], pr[2]))
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt="o", ms=4, color=col, elinewidth=1.1, capsize=0, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(sc_x); ax.set_xticklabels(sc_lab, fontsize=6)
    ax.set_xlabel("Location control: cell fixed effects of size, or paired soundings")
    ax.set_ylabel("Δ median ln $q_c$, 2012–2014 vs Sep 2010–Feb 2011")
    ax.legend(loc="upper left", fontsize=6, title="sand-like layers", title_fontsize=6)
    ax.set_title("The between-period change persists", loc="left"); panel(ax, "a")
    ax = axs[1]
    scales_w = ["cellFE_500m", "cellFE_250m", "cellFE_100m", "cellFE_50m", "pairs_50m_sameprog_pd", "pairs_20m_sameprog_pd"]
    lab_w = ["500 m", "250 m", "100 m", "50 m", "pairs\n≤ 50 m", "pairs\n≤ 20 m"]
    for i, (c, col, lab) in enumerate(st_cols):
        pts = []
        for k, s in enumerate(scales_w):
            v = wr["P3"][c].get(s)
            if v and "slope" in v:
                pts.append((k + (i - 1) * 0.12, v["slope"], v["lo"], v["hi"]))
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt="o", ms=4, color=col, elinewidth=1.1, capsize=0, label=lab)
    ys = tv["archive"]["yang_implied_slope_per_decade"]
    ax.axhline(ys, color="#eb6834", lw=0.8, ls=(0, (4, 2)))
    ax.text(len(scales_w) - 0.6, ys + 0.02, "implied by a 70% rise over 10–800 days", ha="right", va="bottom", fontsize=5.8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(np.arange(len(scales_w))); ax.set_xticklabels(lab_w, fontsize=6)
    ax.set_xlabel("Location control within the Dec 2011–Feb 2016 window")
    ax.set_ylabel("Δ ln $q_c$ per decade of days since 23 Dec 2011")
    ax.set_title("The elapsed-time slope does not", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "B7_location_control")

    # ---- B8: repeat panel
    inc = pd.read_csv(a / "outputs" / "repeat_increments.csv", parse_dates=["date"])
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 66 * MM), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    hl = {"BUR-22": ("#2a78d6", "BUR-22 (Liggins St)"), "AVS-16": ("#eb6834", "AVS-16 (Morris St)")}
    for fam, g in inc.groupby("fam"):
        g = g.sort_values("date")
        col, lab = hl.get(fam, ("#b9b8b0", None))
        ax.plot(g["date"], g["cum_dlogq_1_10"], "o-", ms=3.2, lw=1.1 if fam in hl else 0.8, color=col,
                label=lab, zorder=3 if fam in hl else 2)
        if fam not in hl and g["date"].iloc[-1] > pd.Timestamp("2012-01-01"):
            last = g.iloc[-1]
            ax.text(last["date"] + pd.Timedelta(days=12), last["cum_dlogq_1_10"], fam, fontsize=5.6, color=MUTED, va="center")
    for d, lab in {"2011-02-22": "22 Feb 2011", "2011-06-13": "13 Jun 2011", "2011-12-23": "23 Dec 2011"}.items():
        ax.axvline(pd.Timestamp(d), color=INK, lw=0.6)
        ax.text(pd.Timestamp(d), 1.12, " " + lab, rotation=90, va="top", ha="right", fontsize=5.8, color=INK)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-1.0, 1.15)
    ax.set_xlim(pd.Timestamp("2010-09-01"), pd.Timestamp("2013-02-01"))
    ax.set_ylabel("Cumulative Δ median ln $q_c$, sand-like, 1–10 m")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Eight repeat locations, Oct 2010–Nov 2012", loc="left"); panel(ax, "a")
    ax = axs[1]
    J = inc[inc["from"].notna()].copy()
    J["cls"] = np.where(J["n_events"] == 0, "event-free\n(9–14 days)",
                        np.where(J["to"] == "e", "to Nov 2012\n(505–714 days)", "across one 2011\nevent (87–145 days)"))
    order = ["event-free\n(9–14 days)", "across one 2011\nevent (87–145 days)", "to Nov 2012\n(505–714 days)"]
    bands = [("dlogq_1_4", "1–4 m", "#2a78d6"), ("dlogq_4_8", "4–8 m", "#eb6834"), ("dlogq_8_12", "8–12 m", "#1baf7a")]
    for i, (c, lab, col) in enumerate(bands):
        for k, cl in enumerate(order):
            v = J.loc[J["cls"] == cl, c].dropna().values
            x = k + (i - 1) * 0.22
            ax.scatter(np.full(v.size, x), v, s=9, color=col, lw=0, alpha=0.75, label=lab if k == 0 else None)
            if v.size:
                ax.plot([x - 0.08, x + 0.08], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6)
    ax.set_ylabel("Δ median ln $q_c$ between consecutive visits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6, title="depth band", title_fontsize=6)
    ax.set_title("Changes by interval and depth", loc="left"); panel(ax, "b")
    fig.tight_layout()
    save(fig, outdir / "B8_repeat_panel")


def main(a):
    a = Path(a)
    fig_dir = a / "figures"
    paper_a(a, fig_dir / "paperA")
    paper_b(a, fig_dir / "paperB")
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main(sys.argv[1])
