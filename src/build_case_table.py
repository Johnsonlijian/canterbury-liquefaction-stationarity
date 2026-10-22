"""Build the sounding x event case table for the Canterbury sequence analyses.

Inputs : PRJ-2937 CANTERBURYDATASET.mat (Geyin et al. 2021, DOI 10.17603/ds2-tygh-ht91)
         cpt_dates.csv from decode_cpt_dates.py
Outputs: cases.parquet  one row per sounding x event x profile variant (inverse / measured)
         soundings.parquet  one row per sounding (metadata, event-independent descriptors)

Usage: python -I build_case_table.py <mat> <cpt_dates.csv> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as C  # noqa: E402

EVENTS = {"2010": "Yr2010", "2011": "Yr2011", "2016": "Yr2016"}
EVENT_DATES = {"2010": "2010-09-04", "2011": "2011-02-22", "2016": "2016-02-14"}


def payload(r):
    out = {
        "ID": str(r.ID), "CPTname": str(r.CPTname),
        "E": float(r.EastingNZMG), "N": float(r.NorthingNZMG),
        "lat": float(r.NorthingWGS84), "lon": float(r.EastingWGS84),
        "pd": float(r.pd) if np.ndim(r.pd) == 0 else np.nan,
        "depth": np.atleast_1d(np.asarray(r.depth, float)),
        "qc": np.atleast_1d(np.asarray(r.qc, float)), "fs": np.atleast_1d(np.asarray(r.fs, float)),
        "qc_inv": np.atleast_1d(np.asarray(r.qc_inv, float)), "fs_inv": np.atleast_1d(np.asarray(r.fs_inv, float)),
    }
    for ev, key in EVENTS.items():
        out[f"Mw_{ev}"] = float(getattr(r.Magnitude, key))
        out[f"PGA_{ev}"] = float(getattr(r.PGA, key))
        out[f"PGAsig_{ev}"] = float(getattr(r.PGAsigma, key))
        out[f"GWT_{ev}"] = float(getattr(r.GWT, key))
        out[f"label_{ev}"] = int(getattr(r.Manifestation, key))
    return out


def summarise(i, p):
    rows = []
    summ = {"rec_index": i, "ID": p["ID"], "CPTname": p["CPTname"], "E": p["E"], "N": p["N"],
            "lat": p["lat"], "lon": p["lon"], "pd": p["pd"]}
    for variant, (qk, fk) in {"inv": ("qc_inv", "fs_inv"), "meas": ("qc", "fs")}.items():
        # PRJ-2937 stores qc and fs in kPa
        d, qc, fs = C.clean_profile(p["depth"], p[qk], p[fk])
        if d.size < 10:
            continue
        gamma, valid = C.profile_static(d, qc, fs)
        if variant == "meas":
            summ.update({
                "z_max": float(d[-1]),
                "z_first_valid": float(d[valid][0]) if valid.any() else np.nan,
                "n_points": int(d.size),
            })
        for ev in EVENTS:
            idx, arrs = C.process_event(d, qc, fs, gamma, valid, p[f"GWT_{ev}"], p[f"Mw_{ev}"], p[f"PGA_{ev}"])
            row = {"rec_index": i, "variant": variant, "event": ev,
                   "Mw": p[f"Mw_{ev}"], "PGA": p[f"PGA_{ev}"], "PGAsig": p[f"PGAsig_{ev}"],
                   "GWT": p[f"GWT_{ev}"], "label": p[f"label_{ev}"]}
            row.update(idx)
            if ev == "2011":
                # event-independent-ish soil descriptors from the 2011 effective stress state
                ic, q = arrs["ic"], arrs["qc1ncs"]
                sel = valid & (d <= 10.0)
                selw = sel & (d > p["GWT_2011"])
                row.update({
                    "frac_sandlike_0_10": float(np.mean(ic[sel] < C.IC_LIQ)) if sel.any() else np.nan,
                    "median_Ic_0_10": float(np.median(ic[sel])) if sel.any() else np.nan,
                    "median_q_liq_0_10": float(np.median(q[selw & (ic < C.IC_LIQ)])) if (selw & (ic < C.IC_LIQ)).any() else np.nan,
                    "n_ic_transitions_0_10": int(np.sum(np.abs(np.diff((ic[sel] < C.IC_LIQ).astype(int))))) if sel.sum() > 1 else 0,
                })
            rows.append(row)
    return summ, rows


def main(mat_path, dates_csv, out_dir):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    payloads = [payload(r) for r in recs]
    print(f"loaded {len(payloads)} soundings in {time.time() - t0:.1f}s")
    # Sequential on purpose: workers launched by joblib/loky would not see this
    # script directory under `python -I`, and the full run takes only minutes.
    res = []
    for i, p in enumerate(payloads):
        res.append(summarise(i, p))
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1} soundings ({time.time() - t0:.0f}s)", flush=True)
    sound = pd.DataFrame([s for s, _ in res])
    cases = pd.DataFrame([r for _, rows in res for r in rows])
    dates = pd.read_csv(dates_csv, parse_dates=["date"])
    sound = sound.merge(dates[["rec_index", "date"]], on="rec_index", how="left")
    cases = cases.merge(sound[["rec_index", "ID", "CPTname", "E", "N", "date"]], on="rec_index", how="left")
    cases["event_date"] = pd.to_datetime(cases["event"].map(EVENT_DATES))
    cases["cpt_after_event"] = cases["date"] > cases["event_date"]
    cases["days_cpt_minus_event"] = (cases["date"] - cases["event_date"]).dt.days
    cases["y"] = np.where(cases["label"].between(1, 5), 1, np.where(cases["label"] == 0, 0, np.nan))
    sound.to_parquet(out_dir / "soundings.parquet", index=False)
    cases.to_parquet(out_dir / "cases.parquet", index=False)
    print(f"soundings: {len(sound)}  cases: {len(cases)}  elapsed {time.time() - t0:.1f}s")
    print(cases.groupby(["variant", "event"])[["LPI", "LSN", "LPIish"]].median())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
