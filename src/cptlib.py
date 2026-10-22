"""Vectorised CPT liquefaction-triggering and manifestation-index library.

All coefficients are published values; nothing is fitted to the Canterbury labels.

Triggering: Boulanger & Idriss (2014), CPT procedure (qc1Ncs iteration, CRR_M7.5,
MSF with qc1Ncs-dependent MSFmax, K_sigma, Idriss (1999) r_d).
Soil behaviour type: Robertson (2009) iterative Ic; unit weight Robertson & Cabal (2010).
Fines content: FC = 80 (Ic + Cfc) - 137 (BI2014), Cfc = 0 by default.
Manifestation indices:
  LPI    Iwasaki et al. (1978): int_0^20 F(z) w(z) dz, F = 1 - FS (FS < 1), w = 10 - 0.5 z.
  LPIish Maurer et al. (2015): int_H1^20 F(FS) 25.56 / z dz, F = 1 - FS if FS <= 1 and
         H1 * m(FS) <= 3, m(FS) = exp(5 / (25.56 (1 - FS))) - 1, H1 = non-liquefiable crust.
  LSN    van Ballegooy et al. (2014): 1000 int eps_v / z dz, eps_v (decimal) from
         Zhang et al. (2002) volumetric-strain relations (FS, qc1Ncs).
Units: kPa, m, kN/m^3. Liquefiable: Ic < IC_LIQ, below the water table, qc > 0.
"""
from __future__ import annotations

import numpy as np

PA = 101.325
GW = 9.81
IC_LIQ = 2.6


# ---------------------------------------------------------------- basic stresses
def unit_weight(qt, fs):
    qt = np.asarray(qt, float)
    fs = np.asarray(fs, float)
    rf = np.clip(np.where(qt > 0, fs / np.maximum(qt, 1e-6) * 100.0, 1.0), 0.1, 12.0)
    g = GW * (0.27 * np.log10(rf) + 0.36 * np.log10(np.clip(qt / PA, 1e-3, None)) + 1.236)
    return np.clip(g, 14.0, 22.0)


def stresses(depth, gamma, gwt):
    depth = np.asarray(depth, float)
    dz = np.diff(depth, prepend=0.0)
    dz[0] = depth[0]
    # trapezoidal cumulative integral of gamma with a constant first-segment value
    g_mid = np.concatenate([[gamma[0]], 0.5 * (gamma[1:] + gamma[:-1])])
    sv = np.cumsum(g_mid * dz)
    u = GW * np.maximum(depth - gwt, 0.0)
    sev = np.maximum(sv - u, 0.1)
    return sv, u, sev


def ic_robertson(qt, fs, sv, sev, n_iter=10):
    qnet = np.maximum(np.asarray(qt, float) - sv, 1.0)
    F = np.clip(np.asarray(fs, float) / qnet * 100.0, 0.1, 10.0)
    n = np.full_like(qnet, 0.5)
    ic = np.full_like(qnet, 2.0)
    for _ in range(n_iter):
        Q = np.clip((qnet / PA) * (PA / sev) ** n, 1e-3, None)
        ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = np.clip(0.381 * ic + 0.05 * (sev / PA) - 0.15, 0.0, 1.0)
    return ic


def fines_content(ic, cfc=0.0):
    return np.clip(80.0 * (ic + cfc) - 137.0, 0.0, 100.0)


# ---------------------------------------------------------------- BI2014 triggering
def qc1ncs_bi2014(qc, sev, fc, n_iter=30):
    qcn = np.asarray(qc, float) / PA
    q = np.clip(qcn * 1.0, 1.0, None)
    for _ in range(n_iter):
        m = 1.338 - 0.249 * np.clip(q, 21.0, 254.0) ** 0.264
        cn = np.minimum((PA / sev) ** m, 1.7)
        qc1n = cn * qcn
        dq = (11.9 + qc1n / 14.6) * np.exp(1.63 - 9.7 / (fc + 2.0) - (15.7 / (fc + 2.0)) ** 2)
        q_new = qc1n + dq
        if np.allclose(q_new, q, rtol=1e-6, atol=1e-6):
            q = q_new
            break
        q = q_new
    return q


def rd_idriss(depth, mw):
    z = np.asarray(depth, float)
    a = -1.012 - 1.126 * np.sin(z / 11.73 + 5.133)
    b = 0.106 + 0.118 * np.sin(z / 11.28 + 5.142)
    return np.exp(a + b * mw)


def crr_bi2014(qc1ncs, sev, mw):
    # Cap at 211: beyond this the polynomial grows without bound and such layers are
    # non-liquefiable for any realistic CSR (FS >> 1 either way); avoids overflow.
    q = np.clip(np.asarray(qc1ncs, float), 0.0, 211.0)
    crr75 = np.exp(q / 113.0 + (q / 1000.0) ** 2 - (q / 140.0) ** 3 + (q / 137.0) ** 4 - 2.80)
    msf_max = np.minimum(1.09 + (q / 180.0) ** 3, 2.2)
    msf = 1.0 + (msf_max - 1.0) * (8.64 * np.exp(-mw / 4.0) - 1.325)
    c_sig = np.minimum(1.0 / (37.3 - 8.27 * np.clip(q, None, 211.0) ** 0.264), 0.3)
    k_sig = np.minimum(1.0 - c_sig * np.log(sev / PA), 1.1)
    return crr75 * msf * k_sig


# ---------------------------------------------------------------- Zhang et al. (2002)
_ZHANG_FS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0])


def _zhang_at(fs_node, q):
    q = np.clip(q, 33.0, 200.0)
    if fs_node <= 0.5:
        return 102.0 * q ** -0.82
    if fs_node <= 0.6:
        return np.where(q <= 147.0, 102.0 * q ** -0.82, 2411.0 * q ** -1.45)
    if fs_node <= 0.7:
        return np.where(q <= 110.0, 102.0 * q ** -0.82, 1701.0 * q ** -1.42)
    if fs_node <= 0.8:
        return np.where(q <= 80.0, 102.0 * q ** -0.82, 1690.0 * q ** -1.46)
    if fs_node <= 0.9:
        return np.where(q <= 60.0, 102.0 * q ** -0.82, 1430.0 * q ** -1.48)
    if fs_node <= 1.0:
        return 64.0 * q ** -0.93
    if fs_node <= 1.1:
        return 11.0 * q ** -0.65
    if fs_node <= 1.2:
        return 9.7 * q ** -0.69
    if fs_node <= 1.3:
        return 7.6 * q ** -0.71
    return np.zeros_like(q)


def volumetric_strain_zhang(fs, qc1ncs):
    """Post-liquefaction volumetric strain (percent), linear interpolation in FS."""
    fs = np.asarray(fs, float)
    q = np.asarray(qc1ncs, float)
    nodes = np.stack([_zhang_at(f, q) for f in _ZHANG_FS])  # (10, n)
    fs_c = np.clip(fs, 0.5, 2.0)
    idx = np.clip(np.searchsorted(_ZHANG_FS, fs_c, side="right") - 1, 0, len(_ZHANG_FS) - 2)
    f0 = _ZHANG_FS[idx]
    f1 = _ZHANG_FS[idx + 1]
    w = (fs_c - f0) / (f1 - f0)
    cols = np.arange(q.size)
    e0 = nodes[idx, cols]
    e1 = nodes[idx + 1, cols]
    eps = (1.0 - w) * e0 + w * e1
    eps = np.where(fs <= 0.5, nodes[0, cols], eps)
    return np.where(fs >= 2.0, 0.0, np.clip(eps, 0.0, None))


# ---------------------------------------------------------------- profile processing
def clean_profile(depth, qc, fs):
    depth = np.asarray(depth, float)
    qc = np.asarray(qc, float)
    fs = np.asarray(fs, float)
    ok = np.isfinite(depth) & np.isfinite(qc) & np.isfinite(fs)
    depth, qc, fs = depth[ok], qc[ok], fs[ok]
    order = np.argsort(depth, kind="stable")
    depth, qc, fs = depth[order], qc[order], fs[order]
    keep = np.concatenate([[True], np.diff(depth) > 1e-6])
    return depth[keep], qc[keep], np.clip(fs[keep], 0.0, None)


def profile_static(depth, qc, fs):
    """Event-independent quantities (unit weight uses qc as qt proxy)."""
    valid = qc > 0
    gamma = unit_weight(np.where(valid, qc, 1.0), fs)
    return gamma, valid


def process_event(depth, qc, fs, gamma, valid, gwt, mw, pga, cfc=0.0, zmax=20.0):
    """Per-depth triggering and integrated indices for one sounding and one event."""
    sv, u, sev = stresses(depth, gamma, gwt)
    ic = ic_robertson(np.where(valid, qc, 1.0), fs, sv, sev)
    fc = fines_content(ic, cfc)
    q = qc1ncs_bi2014(np.where(valid, qc, 1.0), sev, fc)
    rd = rd_idriss(depth, mw)
    csr = 0.65 * pga * (sv / sev) * rd
    crr = crr_bi2014(q, sev, mw)
    fs_liq = np.full_like(crr, np.inf, dtype=float)
    np.divide(crr, csr, out=fs_liq, where=csr > 0)
    liq_zone = valid & (ic < IC_LIQ) & (depth > gwt) & (depth <= zmax)
    fs_eff = np.where(liq_zone, fs_liq, np.inf)

    dz = np.gradient(depth) if depth.size > 1 else np.array([0.0])
    dz = np.clip(dz, 0.0, 0.2)
    z = np.maximum(depth, 0.05)

    # LPI
    F = np.where(fs_eff < 1.0, 1.0 - fs_eff, 0.0)
    w = np.clip(10.0 - 0.5 * depth, 0.0, None)
    lpi = float(np.sum(F * w * dz))

    # crust H1: depth to top of first layer with FS < 1 in the liquefiable zone
    trig = np.where(fs_eff < 1.0)[0]
    h1 = float(depth[trig[0]]) if trig.size else float(min(zmax, depth[-1]))

    # LPIish
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        m = np.exp(5.0 / (25.56 * (1.0 - fs_eff))) - 1.0
    f_ish = np.where((fs_eff <= 1.0) & (h1 * m <= 3.0), 1.0 - fs_eff, 0.0)
    f_ish = np.where(depth >= h1, f_ish, 0.0)
    lpiish = float(np.sum(f_ish * 25.56 / z * dz))

    # LSN
    eps = volumetric_strain_zhang(np.where(np.isfinite(fs_eff), fs_eff, 5.0), q) / 100.0
    eps = np.where(liq_zone, eps, 0.0)
    lsn = float(1000.0 * np.sum(eps / z * dz))

    # critical layer (min FS in liquefiable zone, sigma'v >= 10 kPa, z >= 1 m)
    crit_mask = liq_zone & (depth >= max(gwt, 1.0)) & (sev >= 10.0)
    if np.any(crit_mask):
        j = np.where(crit_mask)[0][np.argmin(fs_liq[crit_mask])]
        crit = (float(fs_liq[j]), float(depth[j]), float(q[j]), float(ic[j]))
    else:
        crit = (np.inf, np.nan, np.nan, np.nan)

    # thickness of triggered liquefiable material in 0-10 m
    thick_trig_10 = float(np.sum(dz[(fs_eff < 1.0) & (depth <= 10.0)]))
    return {
        "LPI": lpi, "LPIish": lpiish, "LSN": lsn, "H1": h1,
        "crit_FS": crit[0], "crit_depth": crit[1], "crit_qc1Ncs": crit[2], "crit_Ic": crit[3],
        "thick_trig_10m": thick_trig_10,
    }, {"ic": ic, "qc1ncs": q, "fs_liq": fs_liq, "liq_zone": liq_zone}
