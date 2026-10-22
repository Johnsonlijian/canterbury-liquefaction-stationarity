"""Co-located repeat CPTs at eight eastern-suburb locations (NZGD raw AGS4 exports).

The original soundings (Sep-Dec 2010) were repeated 3-7 days after the 22 February 2011
earthquake (suffix a), 2-3 weeks after it (b, c), 2 days after the 13 June 2011 earthquake (d,
'extra') and in November 2012 (e); the sets are shown qualitatively in Tonkin & Taylor (2015),
Appendix E. Members farther than MAX_SEP from the original are dropped.

Every profile is block-averaged onto a 0.02 m grid. qc1Ncs and Ic follow the IB08 equations used by
Yang et al. (2026) with one fixed water table and demand per family (the PRJ-2937 2011 values of the
original sounding), so differences describe the soil, not the demand. For every ordered pair of
members we compute, in 1-10 m and in depth bands:
  dlogq   median over depths of the change in ln(qc), sand-like points (Ic < 2.6 in both)
  dq1ncs  median change in qc1Ncs over the same points
  dcrit   change in the qc1Ncs at the minimum-FS point (each sounding separately, 1-10 m)
with and without a depth shift (+/-0.3 m) that maximises the correlation of ln(qc).
Pairs are classed by the earthquakes between the two tests: same day, no event, or spanning.
Also printed: the Yang et al. site windows (Liggins St 1.6-7.22 m; Morris St 3.09-4.04 m).

Usage: python -I repeat_panel.py <mat> <outputs_dir>
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from harmonised_metrics import block_average  # noqa: E402
from liqlayer_metrics import crr_ib08, fc_rw98, qc1ncs_ib08  # noqa: E402

EVENTS = [pd.Timestamp(x) for x in ["2010-09-04", "2011-02-22", "2011-06-13", "2011-12-23", "2016-02-14"]]
MAX_SEP = 6.0
ORIG_REC = {"BUR-18": 962, "BUR-22": 5250, "BUR-26": 964, "AVS-16": 820, "AVS-17": 821,
            "SSH-09": 1747, "DAL-09": 1182, "DAL-13": 1185}


def fam_suffix(s):
    s = re.sub(r"\s+", "-", str(s).strip().upper()).replace("CPT-", "").replace("CPT", "")
    m = re.match(r"^-?([A-Z]{3})-?(\d{2})([A-Z]*)$", s)
    return (m.group(1) + "-" + m.group(2), (m.group(3) or "orig").lower()) if m else (None, None)


def process(d, qc, fs, gwt, pga, mw=6.2):
    d, qc, fs = block_average(d, qc, fs)
    gamma, valid = CL.profile_static(d, qc, fs)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    q = qc1ncs_ib08(np.where(valid, qc, 1.0), sev, fc_rw98(ic))
    crr = crr_ib08(q, sev, mw)
    csr = 0.65 * pga * (sv / sev) * CL.rd_idriss(d, mw)
    fsl = np.where(csr > 0, crr / csr, np.inf)
    return pd.DataFrame({"z": np.round(d, 2), "qc": qc, "valid": valid, "ic": ic, "q": q, "fsl": fsl, "below": d > gwt})


def crit(p, lo=1.0, hi=10.0):
    w = p["valid"] & (p["ic"] < 2.6) & p["below"] & (p["z"] >= lo) & (p["z"] <= hi)
    if w.sum() < 5:
        return np.nan
    j = p.loc[w, "fsl"].idxmin()
    return float(p.loc[j, "q"])


def pair_stats(a, b, lo, hi, shift=0.0):
    bb = b.copy()
    bb["z"] = np.round(bb["z"] - shift, 2)
    m = a.merge(bb, on="z", suffixes=("_a", "_b"))
    w = (m["z"] >= lo) & (m["z"] <= hi) & m["valid_a"] & m["valid_b"] & (m["ic_a"] < 2.6) & (m["ic_b"] < 2.6) & m["below_a"]
    if w.sum() < 10:
        return np.nan, np.nan, int(w.sum())
    return (float(np.median(np.log(m.loc[w, "qc_b"]) - np.log(m.loc[w, "qc_a"]))),
            float(np.median(m.loc[w, "q_b"] - m.loc[w, "q_a"])), int(w.sum()))


def best_shift(a, b, lo=1.0, hi=10.0):
    best = (0.0, -np.inf)
    for s in np.round(np.arange(-0.3, 0.301, 0.02), 2):
        bb = b.copy()
        bb["z"] = np.round(bb["z"] - s, 2)
        m = a.merge(bb, on="z", suffixes=("_a", "_b"))
        w = (m["z"] >= lo) & (m["z"] <= hi) & m["valid_a"] & m["valid_b"]
        if w.sum() < 50:
            continue
        r = np.corrcoef(np.log(m.loc[w, "qc_a"]), np.log(m.loc[w, "qc_b"]))[0, 1]
        if r > best[1]:
            best = (float(s), float(r))
    return best


def interval_class(t0, t1):
    if t0.normalize() == t1.normalize():
        return "same day", 0
    n = sum(1 for e in EVENTS if t0 < e <= t1)
    return ("no event" if n == 0 else f"spans {n}"), n


def main(mat_path, out_dir):
    out = Path(out_dir)
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    meta = pd.read_parquet(out / "nzgd_raw_meta.parquet").drop_duplicates("nzgd_id")
    prof = pd.read_parquet(out / "nzgd_raw_profiles.parquet")
    meta[["fam", "suffix"]] = meta["loca_id"].apply(lambda s: pd.Series(fam_suffix(s)))
    meta = meta[meta["fam"].isin(ORIG_REC)]
    members, pairs, windows = [], [], []
    for fam, g in meta.groupby("fam"):
        rec = recs[ORIG_REC[fam]]
        gwt, pga = float(rec.GWT.Yr2011), float(rec.PGA.Yr2011)
        orig = g[g["suffix"] == "orig"].iloc[0]
        g = g.assign(sep=np.hypot(g["E2193"] - orig["E2193"], g["N2193"] - orig["N2193"]))
        g = g[g["sep"] <= MAX_SEP].sort_values("end_date")
        P = {}
        for _, r in g.iterrows():
            p = prof[prof["nzgd_id"] == r["nzgd_id"]].sort_values("depth")
            p = p[np.isfinite(p["depth"]) & np.isfinite(p["qc_kpa"]) & np.isfinite(p["fs_kpa"])]
            d, qc, fs = CL.clean_profile(p["depth"].values, p["qc_kpa"].values, p["fs_kpa"].values)
            P[r["suffix"]] = process(d, qc, fs, gwt, pga)
            pp = P[r["suffix"]]
            members.append({"fam": fam, "suffix": r["suffix"], "nzgd_id": int(r["nzgd_id"]), "date": r["end_date"],
                            "sep_m": float(r["sep"]), "z_end": float(d[-1]), "gwt": gwt, "pga2011": pga,
                            "crit_q_1_10": crit(pp), "med_q_1_10": float(pp.loc[pp["valid"] & (pp["ic"] < 2.6) & pp["below"] & pp["z"].between(1, 10), "q"].median())})
            for lo, hi, tag in [(1.6, 7.22, "liggins"), (3.09, 4.04, "morris")]:
                w = pp["valid"] & pp["z"].between(lo, hi)
                windows.append({"fam": fam, "suffix": r["suffix"], "date": r["end_date"], "window": tag,
                                "mean_q": float(pp.loc[w, "q"].mean()) if w.any() else np.nan,
                                "mean_ic": float(pp.loc[w, "ic"].mean()) if w.any() else np.nan, "n": int(w.sum())})
        rows = g.set_index("suffix")
        for s0, s1 in itertools.combinations(rows.index, 2):
            t0, t1 = rows.loc[s0, "end_date"], rows.loc[s1, "end_date"]
            if t1 < t0:
                s0, s1, t0, t1 = s1, s0, t1, t0
            cls, nev = interval_class(t0, t1)
            sh, r_sh = best_shift(P[s0], P[s1])
            rec_ = {"fam": fam, "from": s0, "to": s1, "t0": t0, "t1": t1, "days": int((t1 - t0).days), "class": cls,
                    "n_events": nev, "sep_m": float(np.hypot(rows.loc[s0, "E2193"] - rows.loc[s1, "E2193"],
                                                           rows.loc[s0, "N2193"] - rows.loc[s1, "N2193"])),
                    "shift_m": sh, "r_logqc": r_sh, "dcrit": crit(P[s1]) - crit(P[s0])}
            for lo, hi in [(1, 10), (1, 4), (4, 8), (8, 12)]:
                for tag, s in [("", 0.0), ("_sh", sh)]:
                    dl, dq, n = pair_stats(P[s0], P[s1], lo, hi, s)
                    rec_.update({f"dlogq_{lo}_{hi}{tag}": dl, f"dq_{lo}_{hi}{tag}": dq, f"n_{lo}_{hi}{tag}": n})
            pairs.append(rec_)
    Mb, Pr, Wn = pd.DataFrame(members), pd.DataFrame(pairs), pd.DataFrame(windows)
    Mb.to_csv(out / "repeat_members.csv", index=False)
    Pr.to_csv(out / "repeat_pairs.csv", index=False)
    Wn.to_csv(out / "repeat_yang_windows.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(Mb.round(2).to_string(index=False))
    print(Wn[(Wn["fam"].isin(["BUR-22"]) & (Wn["window"] == "liggins")) | (Wn["fam"].isin(["AVS-16"]) & (Wn["window"] == "morris"))].round(2).to_string(index=False))
    show = ["fam", "from", "to", "days", "class", "sep_m", "shift_m", "r_logqc", "dlogq_1_10", "dlogq_1_10_sh", "dq_1_10", "dq_1_10_sh", "dcrit"]
    print(Pr[show].round(3).to_string(index=False))
    summ = {}
    for cls, g in Pr.groupby("class"):
        summ[cls] = {"n_pairs": int(len(g)),
                     **{c: [float(g[c].median()), float(g[c].quantile(0.25)), float(g[c].quantile(0.75))]
                        for c in ["dlogq_1_10", "dlogq_1_10_sh", "dq_1_10", "dq_1_10_sh", "dcrit", "dlogq_1_4", "dlogq_4_8", "dlogq_8_12"]}}
    (out / "repeat_panel_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
