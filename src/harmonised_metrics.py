"""Critical-layer and distributional qc1Ncs metrics on harmonised profile geometry.

The Yang et al. (2026) 'liquefiable-layer qc1Ncs' is reproduced by the qc1Ncs at the minimum-FS
point (yang_reproduce.py). A minimum is sensitive to how densely and how far a profile is sampled,
and these differ between programmes. Here every measured profile is processed twice:
  native  original sampling
  h02     block-averaged onto a common 0.02 m grid
and every metric is evaluated in a common window, from max(1 m, water table) to 10 m, with a flag
for profiles that reach 10 m. Procedure: IB08 as printed by Yang et al.; demand D2011 (fixed for all
soundings, so the metric describes the soil, not the demand) and Dprev (as in the reproduction).
Metrics: crit_q (qc1Ncs at minimum FS), p10_q and med_q (10th percentile and median qc1Ncs of
sand-like points, Ic < 2.6), crit_ic, n_sand.

Usage: python -I harmonised_metrics.py <mat> <outputs_dir>
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


def block_average(d, qc, fs, step=0.02):
    k = np.floor(d / step + 1e-9).astype(int)
    uk, inv = np.unique(k, return_inverse=True)
    cnt = np.bincount(inv)
    valid = (qc > 0).astype(float)
    nv = np.bincount(inv, weights=valid)
    qs = np.bincount(inv, weights=qc * valid)
    fsum = np.bincount(inv, weights=fs * valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(nv > 0, qs / nv, 0.0)
        f = np.where(nv > 0, fsum / nv, 0.0)
    return (uk + 0.5) * step, q, f


def metrics(d, qc, fs, gwt, pga, mw):
    gamma, valid = CL.profile_static(d, qc, fs)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    q = qc1ncs_ib08(np.where(valid, qc, 1.0), sev, fc_rw98(ic))
    crr = crr_ib08(q, sev, mw)
    csr = 0.65 * pga * (sv / sev) * CL.rd_idriss(d, mw)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsl = np.where(csr > 0, crr / csr, np.inf)
    win = valid & (ic < 2.6) & (d > gwt) & (d >= 1.0) & (d <= 10.0)
    if win.sum() < 5:
        return {"crit_q": np.nan, "crit_ic": np.nan, "crit_fs": np.nan, "p10_q": np.nan, "med_q": np.nan, "n_sand": int(win.sum())}
    j = np.where(win)[0][np.argmin(fsl[win])]
    return {"crit_q": float(q[j]), "crit_ic": float(ic[j]), "crit_fs": float(fsl[j]),
            "p10_q": float(np.percentile(q[win], 10)), "med_q": float(np.median(q[win])), "n_sand": int(win.sum())}


def main(mat_path, out_dir):
    t0 = time.time()
    out = Path(out_dir)
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    dates = pd.read_parquet(out / "soundings.parquet").set_index("rec_index")["date"]
    rows = []
    for i, r in enumerate(recs):
        d, qc, fs = CL.clean_profile(np.atleast_1d(r.depth), np.atleast_1d(r.qc), np.atleast_1d(r.fs))
        if d.size < 10:
            continue
        dt = dates.get(i, pd.NaT)
        prev = "2010" if (pd.notna(dt) and dt < pd.Timestamp("2011-02-22")) else ("2016" if (pd.notna(dt) and dt >= pd.Timestamp("2016-02-14")) else "2011")
        row = {"rec_index": i, "reach10": bool(d[-1] >= 10.0)}
        for tag, prof in {"native": (d, qc, fs), "h02": block_average(d, qc, fs)}.items():
            for dem in ["2011", "prev"]:
                ev = "2011" if dem == "2011" else prev
                m = metrics(*prof, float(getattr(r.GWT, EV[ev][0])), float(getattr(r.PGA, EV[ev][0])), EV[ev][1])
                row.update({f"{k}_{tag}_D{dem}": v for k, v in m.items()})
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"processed {i + 1} ({time.time() - t0:.0f}s)", flush=True)
    H = pd.DataFrame(rows)
    H.to_parquet(out / "harmonised_metrics.parquet", index=False)
    print(H.describe().T.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
