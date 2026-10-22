"""Create the Paper B graphical abstract for SDEE.

The figure is a deterministic, vector-first summary of the manuscript's four
evidence steps. It does not introduce a causal arrow from elapsed time to
recovery; the final panel preserves the event/ageing identification boundary.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


W, H = 13.28, 5.31  # SDEE graphical-abstract target: 1328 x 531 px at 100 dpi
OUT = Path(__file__).resolve().parents[2] / "manuscripts" / "paperB" / "graphical_abstract"


def box(ax, x, y, w, h, face, edge="#B7C4D0", radius=0.04, lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def add_panel_title(ax, x, title, number):
    ax.text(x, 4.68, number, fontsize=9.8, fontweight="bold", color="#1E5A7A",
            va="center", ha="left")
    ax.text(x + 0.31, 4.68, title, fontsize=9.25, fontweight="bold",
            color="#163042", va="center", ha="left")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.4,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(W, H), dpi=100)
    # Use the complete target canvas. The default Matplotlib subplot margins
    # leave a large unused border and make panel text appear proportionally
    # cramped at the displayed size.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    ax.set_facecolor("#F7FAFC")

    navy = "#163042"
    blue = "#2F78A8"
    teal = "#2B9A9A"
    orange = "#E28F3F"
    red = "#C95B5B"
    grey = "#667685"
    pale_blue = "#EAF3F8"
    pale_orange = "#FFF2E3"
    pale_teal = "#EAF7F5"
    pale_grey = "#EEF2F5"

    # Header and evidence spine.
    ax.text(0.32, 5.12, "Canterbury CPT case histories: dates change what the database can identify",
            fontsize=14.8, fontweight="bold", color=navy, ha="left", va="center")
    ax.text(0.33, 4.88, "A provenance-aware check of discrimination, depth-limited change and elapsed-time claims",
            fontsize=9.35, color=grey, ha="left", va="center")
    ax.plot([0.36, 12.92], [4.50, 4.50], color="#C7D3DC", lw=1.25, zorder=0)

    xs = [0.35, 3.55, 6.75, 9.95]
    pw, ph = 2.90, 3.45
    for x, face in zip(xs, [pale_blue, pale_orange, pale_teal, pale_grey]):
        box(ax, x, 0.90, pw, ph, face, edge="#C9D5DE", radius=0.06, lw=1.0)

    # Panel 1: dates and provenance.
    x = xs[0]
    add_panel_title(ax, x + 0.17, "Date provenance", "01")
    ax.text(x + 0.18, 4.20, "5,667 dates decoded", fontsize=13.0, fontweight="bold", color=blue)
    ax.text(x + 0.18, 3.92, "of 5,668 soundings · 2010–2017", fontsize=8.5, color=grey)
    # timeline
    tx0, ty, tw = x + 0.20, 3.10, 2.48
    ax.plot([tx0, tx0 + tw], [ty, ty], color="#A7B9C7", lw=2.0)
    periods = [(0.00, 0.18, blue, "P0"), (0.18, 0.32, orange, "P1–2"),
               (0.32, 0.92, teal, "P3"), (0.92, 1.00, grey, "P4")]
    for a, b, c, lab in periods:
        ax.add_patch(Rectangle((tx0 + tw * a, ty - 0.12), tw * (b - a), 0.24,
                               facecolor=c, edgecolor="white", linewidth=0.5))
        if b - a > 0.10:
            ax.text(tx0 + tw * (a + b) / 2, ty + 0.28, lab, ha="center", va="bottom",
                    fontsize=8.0, color=navy, fontweight="bold")
    ax.text(tx0, ty - 0.38, "2010", fontsize=7.5, color=grey, ha="left")
    ax.text(tx0 + tw, ty - 0.38, "2017", fontsize=7.5, color=grey, ha="right")
    ax.text(x + 0.18, 2.25, "Programme, coverage and label\nresolution move with the date.",
            fontsize=10.0, color=navy, linespacing=1.25)
    ax.text(x + 0.18, 1.38, "Dates matter", fontsize=10.5,
            fontweight="bold", color=blue)
    ax.text(x + 0.18, 1.13, "measurement context", fontsize=9.0, color=grey)

    # Panel 2: AUC contrast.
    x = xs[1]
    add_panel_title(ax, x + 0.17, "AUC shifts by date", "02")
    ax.text(x + 0.18, 4.18, "2010 Darfield event", fontsize=9.7, color=navy)
    labels = [("early", 0.78, blue), ("2012–13", 0.60, orange)]
    by = [3.48, 2.88]
    for (lab, val, col), y in zip(labels, by):
        ax.text(x + 0.18, y + 0.12, lab, fontsize=8.8, color=grey, va="center")
        ax.add_patch(Rectangle((x + 0.88, y), 1.30 * val / 0.85, 0.30,
                               facecolor=col, edgecolor="none"))
        ax.text(x + 2.16, y + 0.15, f"AUC {val:.2f}", fontsize=9.0,
                fontweight="bold", color=col, va="center", ha="left")
    ax.plot([x + 0.88, x + 0.88 + 1.30], [2.58, 2.58], color="#B5C1CB", lw=0.8)
    for tick, lab in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
        xx = x + 0.88 + 1.30 * tick
        ax.plot([xx, xx], [2.54, 2.62], color="#B5C1CB", lw=0.8)
        ax.text(xx, 2.42, lab, fontsize=7.1, color=grey, ha="center")
    ax.text(x + 0.18, 1.93, "Gap persists with footprint and\n500-m cell controls.",
            fontsize=9.25, color=navy, linespacing=1.25)
    ax.text(x + 0.18, 1.36, "AUC shifts", fontsize=10.0,
            fontweight="bold", color=orange)
    ax.text(x + 0.18, 1.11, "before ground is invoked", fontsize=8.7, color=grey)

    # Panel 3: depth pattern.
    x = xs[2]
    add_panel_title(ax, x + 0.17, "Depth-limited change", "03")
    ax.text(x + 0.18, 4.18, "Matched sites, 2012–14", fontsize=8.9, color=navy)
    rows = [("1–5 m", "+0.39 ln q_c", "≈ +48%", orange),
            ("5–10 m", "+0.14 ln q_c", "smaller", teal),
            ("12–20 m", "≈ +0.01", "no consistent change", grey)]
    yy = [3.53, 3.00, 2.47]
    for (depth, delta, note, col), y in zip(rows, yy):
        ax.text(x + 0.18, y + 0.10, depth, fontsize=8.7, color=grey, va="center")
        ax.add_patch(Rectangle((x + 0.92, y), 1.05 * (0.39 if depth == "1–5 m" else 0.14 if depth == "5–10 m" else 0.03) / 0.39, 0.23,
                               facecolor=col, edgecolor="none"))
        ax.text(x + 2.00, y + 0.11, delta, fontsize=7.8,
                color=col if depth != "12–20 m" else grey, va="center", ha="left")
        ax.text(x + 0.92, y - 0.13, note, fontsize=7.1, color=grey, va="center", ha="left")
    ax.text(x + 0.18, 1.88, "Crust and friction ratio: no\nconsistent change.",
            fontsize=9.25, color=navy, linespacing=1.25)
    ax.text(x + 0.18, 1.36, "Controls localise", fontsize=10.0,
            fontweight="bold", color=teal)
    ax.text(x + 0.18, 1.11, "shallow saturated sand", fontsize=8.7, color=grey)

    # Panel 4: elapsed time and boundary.
    x = xs[3]
    add_panel_title(ax, x + 0.17, "Elapsed time: no law identified", "04")
    ax.text(x + 0.18, 4.18, "Within event-free windows", fontsize=9.7, color=navy)
    xx = [x + 0.30 + 2.20 * i / 5 for i in range(6)]
    yy = [3.40, 3.49, 3.45, 3.52, 3.48, 3.55]
    ax.plot(xx, yy, color=blue, lw=2.0, marker="o", markersize=3.4)
    ax.plot([xx[0], xx[-1]], [3.47, 3.47], color="#C95B5B", lw=1.0, ls="--")
    ax.text(x + 0.18, 3.02, "No positive time trend after\nlocation control",
            fontsize=9.25, color=navy, linespacing=1.25)
    ax.text(x + 0.18, 2.20, "Repeat soundings vary across\nearthquake-spanning intervals.",
            fontsize=9.1, color=navy, linespacing=1.25)
    box(ax, x + 0.18, 1.12, 2.53, 0.62, "#FFF7E8", edge="#E2B46F", radius=0.04)
    ax.text(x + 0.31, 1.59, "Interpretation boundary", fontsize=8.7,
            fontweight="bold", color="#9A5B10")
    ax.text(x + 0.31, 1.30, "sequence change vs ageing\nremains unresolved", fontsize=8.15,
            color="#7C5D32", linespacing=1.0)

    # Bottom takeaway band.
    box(ax, 0.35, 0.19, 12.58, 0.48, "#163042", edge="#163042", radius=0.04, lw=0)
    ax.text(0.58, 0.43, "Takeaway", fontsize=10.2, color="#BFE6F2", fontweight="bold", va="center")
    ax.text(1.83, 0.43, "Record test dates; evaluate case histories by provenance, location and depth.",
            fontsize=9.7, color="white", va="center")

    for ext, dpi in [("png", 100), ("svg", None), ("pdf", None)]:
        path = OUT / f"graphical_abstract.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=None)
        else:
            fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
