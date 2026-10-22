"""Per-sounding candidate definitions of the 'liquefiable-layer qc1Ncs' used by Yang et al. (2026).

Yang et al. describe a liquefiable layer (FS < 1, Ic < 2.6) per sounding and report its qc1Ncs,
Ic, mean depth and thickness (fitted mean thickness 0.5-0.7 m), and they note that "the depth of
minimum FOS" may fall in different strata in successive surveys. Their exact layer rule is not
given, so we compute several candidates and test which reproduces their reported numbers:

  crit   the single point of minimum FS (Ic < 2.6, below the water table, 1 m <= z <= 15 m)
  layer  the contiguous run of FS < 1 points (Ic < 2.6, below the water table, z <= 15 m) that
         contains the minimum-FS point; mean qc1Ncs, mean Ic, mid-depth and thickness
  all    all FS < 1 points with Ic < 2.6 below the water table, z <= 15 m (thickness-weighted)
plus untruncated soil-state descriptors that do not depend on the demand:
  sand_q_med_1_10   median qc1Ncs of sand-like points (Ic < 2.6) below the water table, 1-10 m
  sand_q_med_b{0_5,5_10,10_15}  the same in depth bands

Procedure: Idriss & Boulanger (2008) equations as printed by Yang et al. (IB08) with FC from
Robertson & Wride (1998); BI14 is computed for comparison. Demand options:
  D2011  PGA_2011, GWT_2011, Mw 6.2 for every sounding
  D2010  PGA_2010, GWT_2010, Mw 7.1 for every sounding
  Dprev  the most recent event before the test date (2010 before 22 Feb 2011, 2011 afterwards,
         2016 after 14 Feb 2016)
Profiles: measured (qc, fs) and inverse-filtered (qc_inv, fs_inv).

Usage: python -I yang_metrics.py <mat> <outputs_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from liqlayer_metrics import crr_ib08, fc_rw98, qc1ncs_ib08  # noqa: E402

EV = {"2010": ("Yr2010", 7.1), "2011": ("Yr2011", 6.2), "2016": ("Yr2016", 5.7)}
ZMAX = 15.0


def contiguous_run(mask, j):
    """Indices [a, b] of the run of True values in mask that contains index j."""
    a = j
    while a > 0 and mask[a - 1]:
        a -= 1
    b = j
    while b < mask.size - 1 and mask[b + 1]:
        b += 1
    return a, b


def layer_metrics(d, q, ic, fsl, sand, dz):
    """Return crit / layer / all metrics for one demand."""
    out = {}
    zone = sand & (d <= ZMAX)
    cm = zone & (d >= 1.0)
    if cm.any():
        j = np.where(cm)[0][np.argmin(fsl[cm])]
        out.update({"crit_fs": float(fsl[j]), "crit_q": float(q[j]), "crit_ic": float(ic[j]), "crit_z": float(d[j])})
        liq = zone & (fsl < 1.0)
        if liq[j]:
            a, b = contiguous_run(liq, j)
            sl = slice(a, b + 1)
            w = dz[sl]
            ws = w.sum()
            out.update({"layer_q": float(np.average(q[sl], weights=w)) if ws > 0 else float(q[sl].mean()),
                        "layer_ic": float(np.average(ic[sl], weights=w)) if ws > 0 else float(ic[sl].mean()),
                        "layer_z": float(0.5 * (d[a] + d[b])), "layer_thick": float(max(d[b] - d[a], 0.0) + np.median(dz[sl]))})
        else:
            out.update({"layer_q": np.nan, "layer_ic": np.nan, "layer_z": np.nan, "layer_thick": 0.0})
        if liq.any():
            w = dz[liq]
            ws = w.sum()
            out.update({"all_q": float(np.average(q[liq], weights=w)) if ws > 0 else float(q[liq].mean()),
                        "all_ic": float(np.average(ic[liq], weights=w)) if ws > 0 else float(ic[liq].mean()),
                        "all_z": float(np.average(d[liq], weights=w)) if ws > 0 else float(d[liq].mean()),
                        "all_thick": float(ws)})
            for lo, hi in [(0, 5), (5, 10), (10, 15)]:
                mb = liq & (d >= lo) & (d < hi)
                out[f"all_q_b{lo}_{hi}"] = float(np.mean(q[mb])) if mb.sum() >= 3 else np.nan
        else:
            out.update({"all_q": np.nan, "all_ic": np.nan, "all_z": np.nan, "all_thick": 0.0})
    return out


def one(depth, qc, fs, gwt_by_ev, pga_by_ev, evs):
    d, qc, fs = CL.clean_profile(depth, qc, fs)
    if d.size < 10:
        return None
    gamma, valid = CL.profile_static(d, qc, fs)
    dz = np.clip(np.gradient(d), 0.0, 0.2)
    res = {}
    for ev in evs:
        gwt, pga, mw = gwt_by_ev[ev], pga_by_ev[ev], EV[ev][1]
        sv, u, sev = CL.stresses(d, gamma, gwt)
        ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
        sand = valid & (ic < 2.6) & (d > gwt)
        rd = CL.rd_idriss(d, mw)
        csr = 0.65 * pga * (sv / sev) * rd
        for proc in ["IB08", "BI14"]:
            if proc == "IB08":
                q = qc1ncs_ib08(np.where(valid, qc, 1.0), sev, fc_rw98(ic))
                crr = crr_ib08(q, sev, mw)
            else:
                q = CL.qc1ncs_bi2014(np.where(valid, qc, 1.0), sev, CL.fines_content(ic))
                crr = CL.crr_bi2014(q, sev, mw)
            with np.errstate(divide="ignore", invalid="ignore"):
                fsl = np.where(csr > 0, crr / csr, np.inf)
            m = layer_metrics(d, q, ic, fsl, sand, dz)
            res.update({f"{k}_{proc}_D{ev}": v for k, v in m.items()})
            if ev == "2011":
                # demand-free soil-state descriptors (2011 water table for the effective stress)
                m110 = sand & (d >= 1.0) & (d <= 10.0)
                res[f"sand_q_med_1_10_{proc}"] = float(np.median(q[m110])) if m110.sum() >= 10 else np.nan
                res[f"sand_ic_med_1_10_{proc}"] = float(np.median(ic[m110])) if m110.sum() >= 10 else np.nan
                res[f"sand_n_1_10_{proc}"] = int(m110.sum())
                for lo, hi in [(0, 5), (5, 10), (10, 15)]:
                    mb = sand & (d >= lo) & (d < hi)
                    res[f"sand_q_med_b{lo}_{hi}_{proc}"] = float(np.median(q[mb])) if mb.sum() >= 5 else np.nan
                    res[f"sand_ic_med_b{lo}_{hi}_{proc}"] = float(np.median(ic[mb])) if mb.sum() >= 5 else np.nan
    return res


def main(mat_path, out_dir):
    t0 = time.time()
    out = Path(out_dir)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    rows = []
    for i, r in enumerate(recs):
        gwt = {ev: float(getattr(r.GWT, k)) for ev, (k, _) in EV.items()}
        pga = {ev: float(getattr(r.PGA, k)) for ev, (k, _) in EV.items()}
        row = {"rec_index": i}
        for variant, (qk, fk) in {"meas": ("qc", "fs"), "inv": ("qc_inv", "fs_inv")}.items():
            m = one(np.atleast_1d(np.asarray(r.depth, float)), np.atleast_1d(np.asarray(getattr(r, qk), float)),
                    np.atleast_1d(np.asarray(getattr(r, fk), float)), gwt, pga, ["2010", "2011", "2016"])
            if m is None:
                continue
            row.update({f"{k}_{variant}": v for k, v in m.items()})
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"processed {i + 1} ({time.time() - t0:.0f}s)", flush=True)
    M = pd.DataFrame(rows)
    M.to_parquet(out / "yang_metrics.parquet", index=False)
    print(M.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
