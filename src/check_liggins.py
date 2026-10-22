"""Locate the Liggins Street (CPT-BUR-22 family) and Morris Street soundings used by Yang et al. (2026)
in PRJ-2937 and print depth-interval statistics for comparison with their reported values.

Usage: python -I check_liggins.py <mat> <outputs_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
from liqlayer_metrics import fc_rw98, qc1ncs_ib08, crr_ib08  # noqa: E402


def profile_stats(r, z0, z1, gwt=None, pga=None):
    d, qc, fs = CL.clean_profile(r.depth, r.qc, r.fs)
    gwt = float(r.GWT.Yr2011) if gwt is None else gwt
    pga = float(r.PGA.Yr2011) if pga is None else pga
    gamma, valid = CL.profile_static(d, qc, fs)
    sv, u, sev = CL.stresses(d, gamma, gwt)
    ic = CL.ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    q08 = qc1ncs_ib08(np.where(valid, qc, 1.0), sev, fc_rw98(ic))
    q14 = CL.qc1ncs_bi2014(np.where(valid, qc, 1.0), sev, CL.fines_content(ic))
    m = (d >= z0) & (d <= z1) & valid
    qn = (qc / CL.PA) * (CL.PA / sev) ** 0.5
    return {"n": int(m.sum()), "q_IB08_mean": float(q08[m].mean()), "q_BI14_mean": float(q14[m].mean()),
            "qc_MPa_mean": float(qc[m].mean() / 1000), "qc1N_simple_mean": float(qn[m].mean()),
            "Ic_mean": float(ic[m].mean()), "gwt": gwt, "pga": pga}


def main(mat_path, out_dir):
    out = Path(out_dir)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    link = pd.read_parquet(out / "nzgd_link.parquet")
    snd = pd.read_parquet(out / "soundings.parquet").merge(link, on="rec_index", how="left")
    hits = snd[snd["nzgd_name"].astype(str).str.contains(r"BUR-22|BUR-18|AVS-?1[0-9]|MOR", case=False, regex=True)]
    print(hits[["rec_index", "nzgd_name", "date", "lat", "lon", "pd"]].to_string())
    for _, h in hits.iterrows():
        r = recs[int(h["rec_index"])]
        print(h["nzgd_name"], str(h["date"].date()), profile_stats(r, 1.6, 7.22))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
