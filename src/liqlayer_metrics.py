"""Per-sounding liquefiable-layer metrics in the style of Yang et al. (2026), plus untruncated metrics.

Two procedure variants:
  IB08  Idriss & Boulanger (2008): FC = 1.75 Ic^3.25 - 3.7 (Robertson & Wride 1998),
        dqc1N = (5.4 + qc1N/16) exp(1.63 + 9.7/(FC+0.01) - (15.7/(FC+0.01))^2),
        CRR75 = exp(q/540 + (q/67)^2 - (q/80)^3 + (q/114)^4 - 3), MSF = 6.9 exp(-M/4) - 0.058 <= 1.8.
        This mirrors the equations printed in Yang et al. (2026), Eqs. (1)-(3).
  BI14  Boulanger & Idriss (2014/2016) as in cptlib.
Demand: 2011 Christchurch event (PGA_2011, GWT_2011, Mw 6.2) for every sounding, so the metric
describes the soil state, not the event. Liquefiable points: Ic < 2.6, below the water table,
depth <= 15 m, FS < 1. Per sounding we store the mean qc1Ncs and Ic of liquefiable points, their
mean depth and total thickness (Yang's "liquefiable layer" quantities), and the untruncated median
qc1Ncs of all sand-like points (Ic < 2.6) below the water table in 1-10 m.

Usage: python -I liqlayer_metrics.py <mat> <outputs_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402

PA = 101.325


def fc_rw98(ic):
    ic = np.asarray(ic, float)
    fc = 1.75 * ic ** 3.25 - 3.7
    fc = np.where(ic < 1.26, 0.0, fc)
    return np.clip(fc, 0.0, 100.0)


def qc1ncs_ib08(qc, sev, fc, n_iter=30):
    qcn = np.asarray(qc, float) / PA
    q = np.clip(qcn, 1.0, None)
    for _ in range(n_iter):
        m = 1.338 - 0.249 * np.clip(q, 21.0, 254.0) ** 0.264
        cn = np.minimum((PA / sev) ** m, 1.7)
        qc1n = cn * qcn
        dq = (5.4 + qc1n / 16.0) * np.exp(1.63 + 9.7 / (fc + 0.01) - (15.7 / (fc + 0.01)) ** 2)
        q_new = qc1n + dq
        if np.allclose(q_new, q, rtol=1e-6, atol=1e-6):
            q = q_new
            break
        q = q_new
    return q


def crr_ib08(q, sev, mw):
    q = np.clip(np.asarray(q, float), 0.0, 211.0)
    crr75 = np.exp(q / 540.0 + (q / 67.0) ** 2 - (q / 80.0) ** 3 + (q / 114.0) ** 4 - 3.0)
    msf = min(6.9 * np.exp(-mw / 4.0) - 0.058, 1.8)
    c_sig = np.minimum(1.0 / (37.3 - 8.27 * np.clip(q, None, 211.0) ** 0.264), 0.3)
    k_sig = np.minimum(1.0 - c_sig * np.log(sev / PA), 1.1)
    return crr75 * msf * k_sig


def metrics_one(depth, qc, fs, gwt, pga, mw, variant):
    d, qc, fs = CL.clean_profile(depth, qc, fs)
    if d.size < 10:
        return None
    gamma, valid = CL.profile_static(d, qc, fs)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    if variant == "IB08":
        fc = fc_rw98(ic)
        q = qc1ncs_ib08(np.where(valid, qc, 1.0), sev, fc)
        crr = crr_ib08(q, sev, mw)
    else:
        fc = CL.fines_content(ic)
        q = CL.qc1ncs_bi2014(np.where(valid, qc, 1.0), sev, fc)
        crr = CL.crr_bi2014(q, sev, mw)
    rd = CL.rd_idriss(d, mw)
    csr = 0.65 * pga * (sv / sev) * rd
    with np.errstate(divide="ignore", invalid="ignore"):
        fsl = np.where(csr > 0, crr / csr, np.inf)
    dz = np.clip(np.gradient(d), 0.0, 0.2) if d.size > 1 else np.array([0.0])
    sand = valid & (ic < 2.6) & (d > gwt)
    liq = sand & (d <= 15.0) & (fsl < 1.0)
    out = {}
    if liq.any():
        w = dz[liq]
        out.update({"liq_q_mean": float(np.average(q[liq], weights=w)) if w.sum() > 0 else float(np.mean(q[liq])),
                    "liq_ic_mean": float(np.average(ic[liq], weights=w)) if w.sum() > 0 else float(np.mean(ic[liq])),
                    "liq_depth_mean": float(np.average(d[liq], weights=w)) if w.sum() > 0 else float(np.mean(d[liq])),
                    "liq_thick": float(w.sum()), "liq_minFS": float(np.min(fsl[liq]))})
    else:
        out.update({"liq_q_mean": np.nan, "liq_ic_mean": np.nan, "liq_depth_mean": np.nan, "liq_thick": 0.0,
                    "liq_minFS": float(np.min(np.where(sand & (d <= 15.0), fsl, np.inf))) if (sand & (d <= 15.0)).any() else np.inf})
    m110 = sand & (d >= 1.0) & (d <= 10.0)
    out["sand_q_median_1_10"] = float(np.median(q[m110])) if m110.sum() >= 10 else np.nan
    out["sand_ic_median_1_10"] = float(np.median(ic[m110])) if m110.sum() >= 10 else np.nan
    for lo, hi in [(0, 5), (5, 10), (10, 15)]:
        mb = liq & (d >= lo) & (d < hi)
        out[f"liq_q_mean_{lo}_{hi}"] = float(np.mean(q[mb])) if mb.sum() >= 3 else np.nan
    return out


def main(mat_path, out_dir):
    out = Path(out_dir)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    rows = []
    for i, r in enumerate(recs):
        base = {"rec_index": i, "pd": float(r.pd) if np.ndim(r.pd) == 0 else np.nan}
        for variant in ["IB08", "BI14"]:
            m = metrics_one(r.depth, r.qc, r.fs, float(r.GWT.Yr2011), float(r.PGA.Yr2011), 6.2, variant)
            if m is None:
                continue
            base.update({f"{k}_{variant}": v for k, v in m.items()})
        rows.append(base)
        if (i + 1) % 1000 == 0:
            print("processed", i + 1, flush=True)
    M = pd.DataFrame(rows)
    M.to_parquet(out / "liqlayer_metrics.parquet", index=False)
    print(M.describe().T.head(30))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
