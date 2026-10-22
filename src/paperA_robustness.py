"""Robustness and spatial-transfer analyses for Paper A (site memory).

1. Fines-content correlation. Indices recomputed with CFC = -0.29 and +0.29 in
   FC = 80 (Ic + CFC) - 137 (about one standard deviation of the fitting parameter in
   Boulanger and Idriss 2014), inverse-filtered profiles; three-event site-term model and joint
   state-dependence model refitted. A regional bias in the fines correlation shifts every index;
   if the site term were such a bias it would change materially.
2. Outcome threshold. Outcome = moderate or worse manifestation (codes 2-5) against none or minor
   (0-1), primary indices.
3. Neighbour-only forecast. For each held-out event, site terms are estimated from the other two
   events; each site in the held-out event is scored with b_bar x + w * d_nb, where d_nb is the
   inverse-distance-weighted mean of the site terms of other sites between 20 m and R (100 or
   250 m) away, so that no co-located history is used. AUC gain over the index alone, with
   1-km block bootstrap intervals; w = 1 and the 2016 shrinkage w = 0.25.

Usage: python -I paperA_robustness.py <mat> <outputs_dir> [part ...]   (parts: cfc threshold neighbour)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptlib as CL  # noqa: E402
import state_dependence as SD  # noqa: E402
from site_terms_glmm import auc, block_boot_gain, eb_from_params, fit  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")
EVK = {"2010": "Yr2010", "2011": "Yr2011", "2016": "Yr2016"}


def base_cases(out, fname="cases.parquet"):
    c = pd.read_parquet(out / fname)
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    c["block"] = (np.floor(c["E"] / 1000).astype(int) * 100000 + np.floor(c["N"] / 1000).astype(int))
    return c


def cfc_indices(mat_path, out, cfcs=(-0.29, 0.29)):
    f = out / "indices_cfc.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t0 = time.time()
    recs = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["CANTERBURYDATASET"]
    rows = []
    for i, r in enumerate(recs):
        d, qc, fs = CL.clean_profile(np.atleast_1d(r.depth), np.atleast_1d(r.qc_inv), np.atleast_1d(r.fs_inv))
        if d.size < 10:
            continue
        gamma, valid = CL.profile_static(d, qc, fs)
        for ev, k in EVK.items():
            for cfc in cfcs:
                idx, _ = CL.process_event(d, qc, fs, gamma, valid, float(getattr(r.GWT, k)), float(getattr(r.Magnitude, k)),
                                          float(getattr(r.PGA, k)), cfc=cfc)
                rows.append({"rec_index": i, "event": ev, "cfc": cfc, "LSN": idx["LSN"], "LPI": idx["LPI"], "LPIish": idx["LPIish"]})
        if (i + 1) % 1000 == 0:
            print(f"cfc indices: {i + 1} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(f, index=False)
    return R


def site_and_sd(c, ix):
    c = c.copy()
    c["x"] = np.log1p(c[ix])
    o, _, _, _ = fit(c, ["2010", "2011", "2016"], epoch_split_2010=True)
    W = SD.build(c, ix)
    r0, _ = SD.fit(W, ix, "none")
    r1, se1 = SD.fit(W, ix, "common", start=r0.x)
    return {"tau": o["tau"], "ICC": o["ICC"], "n_rows": o["n_rows"], "g_common": float(r1.x[10]),
            "g_common_se": se1.get(10), "LR_common": float(2 * (r0.fun - r1.fun))}


def part_cfc(mat_path, out):
    c0 = base_cases(out)
    R = cfc_indices(mat_path, out)
    res = {}
    for cfc in sorted(R["cfc"].unique()):
        ri = R[R["cfc"] == cfc].drop(columns="cfc")
        c = c0.drop(columns=["LSN", "LPI", "LPIish"]).merge(ri, on=["rec_index", "event"], how="inner")
        res[f"cfc={cfc:+.2f}"] = {}
        for ix in ["LSN", "LPI", "LPIish"]:
            r = site_and_sd(c, ix)
            # discrimination by event for context
            r["AUC_by_event"] = {ev: auc(g["y"], g[ix]) for ev, g in c.groupby("event")}
            res[f"cfc={cfc:+.2f}"][ix] = r
            print("cfc", cfc, ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "AUC_by_event"}, flush=True)
    return res


def part_threshold(out):
    c = base_cases(out)
    c["y"] = np.where(c["label"].between(2, 5), 1, np.where(c["label"].between(0, 1), 0, np.nan))
    c = c[c["y"].notna()].copy()
    res = {"base_rate": c.groupby("event")["y"].mean().round(4).to_dict(), "n_pos": c.groupby("event")["y"].sum().astype(int).to_dict()}
    for ix in ["LSN", "LPI", "LPIish"]:
        r = site_and_sd(c, ix)
        res[ix] = r
        print("moderate+", ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return res


def neighbour_mean(xy_target, xy_source, d_source, rmin, rmax):
    tree = cKDTree(xy_source)
    out = np.zeros(len(xy_target))
    nn = np.zeros(len(xy_target), int)
    for i, p in enumerate(xy_target):
        idx = tree.query_ball_point(p, rmax)
        if not idx:
            continue
        dist = np.hypot(*(xy_source[idx] - p).T)
        m = dist >= rmin
        if not m.any():
            continue
        w = 1.0 / dist[m]
        out[i] = float(np.sum(w * d_source[np.array(idx)[m]]) / np.sum(w))
        nn[i] = int(m.sum())
    return out, nn


def part_neighbour(out):
    c = base_cases(out)
    geo = pd.read_parquet(out / "soundings_geo.parquet")[["rec_index", "X2193", "Y2193"]]
    res = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        c["x"] = np.log1p(c[ix])
        res[ix] = []
        for h in ["2010", "2011", "2016"]:
            train = [e for e in ["2010", "2011", "2016"] if e != h]
            o, eb, theta, names = fit(c, train, epoch_split_2010=True)
            other = c[c["event"].isin(train)]
            ebh = eb_from_params(other, theta, names).merge(geo, on="rec_index")
            d = c[c["event"] == h].merge(geo, on="rec_index").merge(ebh[["rec_index", "d_hat"]], on="rec_index", how="left")
            bbar = float(np.mean([o["b"][k] for k in names]))
            d["s_erg"] = d["x"]
            d["s_own"] = bbar * d["x"] + d["d_hat"].fillna(0.0)
            row = {"held_out": h, "n": int(len(d)), "AUC_ergodic": auc(d["y"], d["s_erg"]), "AUC_own": auc(d["y"], d["s_own"])}
            g, ci, _ = block_boot_gain(d, "s_erg", "s_own", n_boot=1000)
            row["gain_own"] = [g, ci]
            for rmax in [100, 250]:
                dn, nn = neighbour_mean(d[["X2193", "Y2193"]].values, ebh[["X2193", "Y2193"]].values, ebh["d_hat"].values, 20.0, rmax)
                d[f"dnb{rmax}"] = dn
                row[f"share_with_neighbours_{rmax}m"] = float(np.mean(nn > 0))
                row[f"corr_dnb_own_{rmax}m"] = float(np.corrcoef(dn[nn > 0], d["d_hat"].fillna(0).values[nn > 0])[0, 1])
                for w in ([1.0, 0.25] if h == "2016" else [1.0, 0.5]):
                    col = f"s_nb{rmax}_w{w}"
                    d[col] = bbar * d["x"] + w * d[f"dnb{rmax}"]
                    g, ci, _ = block_boot_gain(d, "s_erg", col, n_boot=1000)
                    row[f"AUC_nb{rmax}_w{w}"] = auc(d["y"], d[col])
                    row[f"gain_nb{rmax}_w{w}"] = [g, ci]
            res[ix].append(row)
            print(ix, h, {k: v for k, v in row.items()}, flush=True)
    return res


def part_epoch(out):
    """Site term and state dependence for sites sounded before 22 Feb 2011 (profiles that predate
    the 2011 and 2016 earthquakes) and for sites sounded in 2012-2013 (profiles that postdate the
    2010 and 2011 earthquakes). If post-event changes in the profiles created the site term, it
    would be much smaller in the first group."""
    c = base_cases(out)
    P = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "period"]]
    c = c.merge(P, on="rec_index", how="left")
    res = {}
    for grp, sel in {"sounded_before_22Feb2011": c["period"] == "P0", "sounded_2012_2013": c["period"] == "P3"}.items():
        cc = c[sel].copy()
        res[grp] = {"n_sites": int(cc["rec_index"].nunique()), "base_rate": cc.groupby("event")["y"].mean().round(3).to_dict()}
        for ix in ["LSN", "LPI", "LPIish"]:
            cc["x"] = np.log1p(cc[ix])
            # P0 sites all share the same 2010 epoch, so no epoch split is needed within a group
            o, _, _, _ = fit(cc, ["2010", "2011", "2016"], epoch_split_2010=False)
            o2, _, _, _ = fit(cc, ["2011", "2016"], epoch_split_2010=False)
            W = SD.build(cc, ix)
            W["p0"] = 0
            r0, _ = SD.fit(W, ix, "none")
            r1, _ = SD.fit(W, ix, "common", start=r0.x)
            # the second 2010 equation (indices 1 and 3) is unused when every site shares one
            # epoch, so the standard error of g is taken from the Hessian of the identified block
            keep = [0, 2, 4, 5, 6, 7, 8, 9, 10]
            h = 1e-4
            H = np.zeros((len(keep), len(keep)))
            for a_, i in enumerate(keep):
                for b_, j in enumerate(keep):
                    if b_ < a_:
                        continue
                    ei = np.zeros(11); ej = np.zeros(11); ei[i] = h; ej[j] = h
                    val = (SD.nll(r1.x + ei + ej, W, ix, "common") - SD.nll(r1.x + ei - ej, W, ix, "common")
                           - SD.nll(r1.x - ei + ej, W, ix, "common") + SD.nll(r1.x - ei - ej, W, ix, "common")) / (4 * h * h)
                    H[a_, b_] = H[b_, a_] = val
            se_g = float(np.sqrt(max(np.linalg.inv(H)[-1, -1], 0.0)))
            res[grp][ix] = {"tau_3ev": o["tau"], "ICC_3ev": o["ICC"], "LR_3ev": o["LR_vs_tau0"], "tau_2011_2016": o2["tau"],
                            "LR_2011_2016": o2["LR_vs_tau0"], "g_common": float(r1.x[10]), "g_common_se": se_g,
                            "LR_g": float(2 * (r0.fun - r1.fun)), "n_sd": int(len(W))}
            print(grp, ix, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[grp][ix].items()}, flush=True)
    return res


def main(mat_path, out_dir, parts):
    out = Path(out_dir)
    f = out / "paperA_robustness.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    if "epoch" in parts:
        res["epoch"] = part_epoch(out)
    if "cfc" in parts:
        res["cfc"] = part_cfc(mat_path, out)
    if "threshold" in parts:
        res["threshold_moderate_plus"] = part_threshold(out)
    if "neighbour" in parts:
        res["neighbour_forecast"] = part_neighbour(out)
    f.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["cfc", "threshold", "neighbour"])
