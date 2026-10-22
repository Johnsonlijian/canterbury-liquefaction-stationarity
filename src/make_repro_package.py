"""Assemble a local reproducibility package with SHA-256 checksums (not uploaded anywhere).

Layout of the package (zip and folder under <base>/reproducibility/; names are kept short so that
the package unpacks within the Windows path-length limit):
  scripts/            all analysis scripts
  outputs/            JSON summaries and small CSV tables used in the manuscripts
                      (outputs/snapshot_allsites/: summaries before the ground-improved soundings
                      were excluded)
  licence_check/      per-sounding tables derived from PRJ-2937 (decoded dates, indices, strata) and
                      tables derived from NZGD exports; release only after checking the DesignSafe
                      licence and the NZGD terms of use
  figures/            PDF/SVG/PNG figures of both papers
  manuscripts/        LaTeX sources, macro files, bibliography, compiled PDFs and the build script
  MANIFEST_SHA256.txt checksum of every file
Excluded: raw PRJ-2937 file, raw NZGD exports and parsed NZGD profiles, NZGD catalogue linkage,
GNS GM3 polygons (re-downloaded by fetch_gm3_units.py), LaTeX auxiliary files, logs.

Usage: python -I make_repro_package.py <base_dir> <tag>
"""
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

EXCLUDE_OUTPUTS = {"nzgd_raw_profiles.parquet", "nzgd_raw_meta.parquet", "nzgd_link.parquet"}
# per-sounding tables derived from PRJ-2937 (DesignSafe licence)
PER_SOUNDING = {"cases.parquet", "cases_all.parquet", "soundings.parquet", "soundings_geo.parquet", "wide_inv.parquet",
                "cpt_dates.csv", "liqlayer_metrics.parquet", "yang_metrics.parquet", "harmonised_metrics.parquet",
                "negative_controls_strata.parquet", "strata_variants.parquet", "profile_geometry.parquet",
                "programme_classes.parquet", "depth_band_stats.parquet", "indices_cfc.parquet",
                "site_terms_eb_LSN.parquet", "site_terms_eb_LPI.parquet", "site_terms_eb_LPIish.parquet", "timing_close_pairs.csv"}
# tables derived from NZGD exports (NZGD terms of use)
NZGD_DERIVED = {"repeat_members.csv", "repeat_pairs.csv", "repeat_increments.csv", "repeat_yang_windows.csv",
                "preces_pairs.csv", "raw_unit_check.csv"}
TEX_KEEP = {".tex", ".bib", ".pdf", ".txt", ".ps1"}
LICENCE = "licence_check"


def L(p):
    """Path string usable beyond the 260-character limit on Windows."""
    s = os.path.abspath(str(p))
    return "\\\\?\\" + s if os.name == "nt" and len(s) >= 240 and not s.startswith("\\\\?\\") else s


def copy(src, target):
    os.makedirs(L(Path(target).parent), exist_ok=True)
    shutil.copy2(L(src), L(target))


def copy_outputs(src, dst, prefix=""):
    """Summaries to outputs/, per-sounding and NZGD-derived tables to licence_check/."""
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.name in EXCLUDE_OUTPUTS or p.suffix in (".log", ".err"):
            continue
        sub = LICENCE if (p.name in PER_SOUNDING or p.name in NZGD_DERIVED) else "outputs"
        copy(p, dst / sub / prefix / p.name)


def sha256(p):
    h = hashlib.sha256()
    with open(L(p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(base, tag):
    base = Path(base)
    dst = base / "reproducibility" / f"P1P3_repro_{tag}"
    if dst.exists():
        shutil.rmtree(L(dst))
    for p in sorted((base / "analysis" / "scripts").glob("*.py")):
        copy(p, dst / "scripts" / p.name)
    out = base / "analysis" / "outputs"
    copy_outputs(out, dst)
    if (out / "snapshot_allsites").is_dir():
        copy_outputs(out / "snapshot_allsites", dst, "snapshot_allsites")
    for p in sorted((base / "analysis" / "figures").rglob("*")):
        if p.is_file():
            copy(p, dst / "figures" / p.relative_to(base / "analysis" / "figures"))
    for paper in ["paperA", "paperB", "shared"]:
        src = base / "manuscripts" / paper
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in TEX_KEEP and "build_" not in p.name:
                copy(p, dst / "manuscripts" / p.relative_to(base / "manuscripts"))
    for name in ["build_manuscripts.ps1", "check_macro_spaces.py"]:
        copy(base / "manuscripts" / name, dst / "manuscripts" / name)
    copy(base / "README.md", dst / "README_project.md")
    with open(L(dst / "README_reproducibility.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Reproducibility package (local; not released)\n\n"
            "Inputs not included: PRJ-2937 `CANTERBURYDATASET.mat` (DesignSafe, DOI 10.17603/ds2-tygh-ht91); NZGD AGS4 "
            "exports (New Zealand Geotechnical Database, terms of use); GNS GM3 polygons (fetched by `fetch_gm3_units.py`).\n\n"
            "Environment: Python 3.11 with numpy, scipy, pandas, pyarrow, scikit-learn, statsmodels, shapely, pyproj, "
            "matplotlib; TeX Live/MiKTeX with elsarticle.\n\n"
            "Order: decode_cpt_dates -> build_case_table -> link_nzgd_metadata -> programme_classes -> analyses (see "
            "`README_project.md`) -> make_figures -> gen_si_paperB -> build_manuscripts.ps1 (which runs gen_tex_numbers, "
            "paperA_numbers, paperB_numbers, paperB_prov_numbers and check_macro_spaces).\n\n"
            "`outputs/snapshot_allsites/` holds the summaries computed before the 122 ground-improved soundings were "
            "excluded; the 'including ground-improved sites' row of Paper A's sensitivity table uses them.\n\n"
            f"`{LICENCE}/` holds per-sounding tables derived from PRJ-2937 and, for repeat_*.csv, preces_pairs.csv and "
            "raw_unit_check.csv, tables derived from NZGD exports. Check the DesignSafe licence and the NZGD terms of use "
            "before any public release.\n")
    files = sorted(p for p in dst.rglob("*") if p.is_file())
    with open(L(dst / "MANIFEST_SHA256.txt"), "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(dst).as_posix()}\n")
    zpath = dst.with_suffix(".zip")
    if zpath.exists():
        os.remove(L(zpath))
    longest = 0
    with zipfile.ZipFile(L(zpath), "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dst.rglob("*")):
            if p.is_file():
                arc = p.relative_to(dst.parent).as_posix()
                longest = max(longest, len(arc))
                z.write(L(p), arc)
    print(f"package: {dst}\nfiles: {len(files) + 1}  zip: {zpath.stat().st_size / 1e6:.1f} MB  "
          f"longest path in zip: {longest} characters\nsha256(zip): {sha256(zpath)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
