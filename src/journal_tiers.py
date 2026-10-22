"""Look up CAS-2025 (upgraded), XR-2026 and JCR-2025 tiers for candidate journals.

Reads the local partition tables in W:\\01_PROJECTS\\NAS_DRIVE\\IMUT\\期刊分区表 and
writes a compact CSV for the target-journal ladder. Exact (case-insensitive)
title match first, then a normalised match (ampersand/and, punctuation).

Usage: python -I journal_tiers.py <tier_dir> <out_csv>
"""
import re
import sys
from pathlib import Path

import pandas as pd

CANDIDATES = [
    "Engineering Geology",
    "Soil Dynamics and Earthquake Engineering",
    "Earthquake Spectra",
    "Journal of Geotechnical and Geoenvironmental Engineering",
    "Geotechnique",
    "Canadian Geotechnical Journal",
    "Computers and Geotechnics",
    "Acta Geotechnica",
    "Georisk-Assessment and Management of Risk for Engineered Systems and Geohazards",
    "Journal of Rock Mechanics and Geotechnical Engineering",
    "Bulletin of Earthquake Engineering",
    "Natural Hazards and Earth System Sciences",
    "Computer-Aided Civil and Infrastructure Engineering",
    "Reliability Engineering & System Safety",
    "Structural Safety",
    "Soils and Foundations",
    "Journal of Earthquake Engineering",
    "Earthquake Engineering & Structural Dynamics",
    "Bulletin of the Seismological Society of America",
    "International Journal of Disaster Risk Reduction",
    "Natural Hazards",
    "Communications Earth & Environment",
    "npj Natural Hazards",
    "Geophysical Research Letters",
    "Engineering",
    "ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems Part A-Civil Engineering",
    "Probabilistic Engineering Mechanics",
    "International Journal for Numerical and Analytical Methods in Geomechanics",
    "Underground Space",
    "Transportation Geotechnics",
    "Earth-Science Reviews",
    "Geoscience Frontiers",
    "Journal of Geophysical Research-Solid Earth",
    "Earthquake Engineering and Engineering Vibration",
    "Data-Centric Engineering",
    "Geomechanics for Energy and the Environment",
    "Landslides",
    "Earth Surface Processes and Landforms",
    "Quaternary Science Reviews",
    "Sedimentology",
]


def norm(s):
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lookup(df, col, name):
    hit = df[df[col].astype(str).str.lower() == name.lower()]
    if hit.empty:
        n = norm(name)
        hit = df[df[col].map(norm) == n]
    return hit


def main(tier_dir, out_csv):
    d = Path(tier_dir)
    cas = pd.read_csv(d / "中科院分区表2025升级版_FQBJCR2025.csv", encoding="utf-8", dtype=str)
    xr = pd.read_csv(d / "新锐期刊分区表2026_XR2026.csv", encoding="utf-8", dtype=str)
    jcr = pd.read_csv(d / "JCR2025.csv", encoding="utf-8", dtype=str)
    rows = []
    for name in CANDIDATES:
        c = lookup(cas, "Journal", name)
        x = lookup(xr, "Journal", name)
        j = lookup(jcr, "Journal", name)
        row = {"journal": name}
        if not c.empty:
            r = c.iloc[0]
            row.update({
                "cas2025_major": r.get("大类"), "cas2025_zone": r.get("大类分区"), "cas2025_top": r.get("Top"),
                "cas2025_minor1": r.get("小类1"), "cas2025_minor1_zone": r.get("小类1分区"),
                "cas2025_oa": r.get("Open Access"),
            })
        if not x.empty:
            r = x.iloc[0]
            row.update({"xr2026_major": r.get("大类中文名"), "xr2026_zone": r.get("大类新锐分区"), "xr2026_top": r.get("Top")})
        if not j.empty:
            r = j.iloc[0]
            row.update({"jcr2025_if": r.get("IF(2025)"), "jcr2025_cat1": r.get("Category_1"), "jcr2025_q1": r.get("IF Quartile(2025)_1")})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_rows", 200, "display.width", 250, "display.max_colwidth", 40):
        print(out[["journal", "cas2025_zone", "cas2025_top", "xr2026_zone", "xr2026_top", "jcr2025_if", "jcr2025_q1"]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
