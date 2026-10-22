"""Join GM3 geomorphic units to soundings and relate site terms to mapped geomorphology.

Outputs
  soundings_geo.parquet   rec_index + GM3 environ/simplename/age (smallest containing unit)
  geomorph_site_terms.json  site-term summaries by environment and landform, variance
                            explained by mapped units (with/without CPT descriptors), and
                            distance-to-mapped-channel effects (GM3 'river channel' and
                            fluvial-channel units used as free-face proxies).

Usage: python -I join_geomorphology.py <outputs_dir> <gm3_geojson>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


def main(out_dir, gm3_path):
    out = Path(out_dir)
    with open(gm3_path, encoding="utf-8") as fh:
        gj = json.load(fh)
    to_nztm = Transformer.from_crs(4326, 2193, always_xy=True)
    polys, props = [], []
    from shapely.ops import transform as stransform
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        geom = shape(g)
        geom = stransform(lambda x, y, z=None: to_nztm.transform(x, y), geom)
        if not geom.is_valid:
            geom = geom.buffer(0)
        polys.append(geom)
        props.append(f["properties"])
    areas = np.array([p.area for p in polys])
    tree = STRtree(polys)
    snd = pd.read_parquet(out / "soundings.parquet")
    X, Y = to_nztm.transform(snd["lon"].values, snd["lat"].values)
    snd["X2193"], snd["Y2193"] = X, Y
    env, lf, amin, amax, name = [], [], [], [], []
    for x, y in zip(X, Y):
        pt = Point(x, y)
        cand = [i for i in tree.query(pt) if polys[i].contains(pt)]
        if not cand:
            env.append(None); lf.append(None); amin.append(np.nan); amax.append(np.nan); name.append(None)
            continue
        i = min(cand, key=lambda j: areas[j])
        p = props[i]
        env.append(p.get("environ")); lf.append(p.get("simplename")); name.append(p.get("name"))
        amin.append(pd.to_numeric(p.get("absmin_yrs"), errors="coerce"))
        amax.append(pd.to_numeric(p.get("absmax_yrs"), errors="coerce"))
    snd["gm3_environ"] = env
    snd["gm3_landform"] = lf
    snd["gm3_name"] = name
    snd["gm3_age_mid_yrs"] = (np.array(amin, float) + np.array(amax, float)) / 2

    # distance to mapped channel-type units (free-face proxy)
    chan_idx = [i for i, p in enumerate(props)
                if any(k in str(p.get("simplename", "")).lower() for k in ["channel", "river", "stream", "estuar"])
                or "channel" in str(p.get("name", "")).lower()]
    chan_geoms = [polys[i].boundary for i in chan_idx]
    if chan_geoms:
        ctree = STRtree(chan_geoms)
        dist = []
        for x, y in zip(X, Y):
            pt = Point(x, y)
            j = ctree.nearest(pt)
            dist.append(pt.distance(chan_geoms[j]))
        snd["dist_channel_m"] = dist
    snd[["rec_index", "X2193", "Y2193", "gm3_environ", "gm3_landform", "gm3_name", "gm3_age_mid_yrs"] +
        (["dist_channel_m"] if chan_geoms else [])].to_parquet(out / "soundings_geo.parquet", index=False)

    res = {"n_polygons": len(polys), "n_channel_units": len(chan_idx),
           "matched_fraction": float(np.mean(pd.notna(snd["gm3_environ"]))),
           "environ_counts": snd["gm3_environ"].value_counts().to_dict(),
           "landform_counts_top25": snd["gm3_landform"].value_counts().head(25).to_dict()}

    eb = pd.read_parquet(out / "site_terms_eb_LSN.parquet")
    D = eb.merge(snd, on="rec_index")
    c11 = pd.read_parquet(out / "cases.parquet")
    c11 = c11[(c11["variant"] == "inv") & (c11["event"] == "2011")][
        ["rec_index", "GWT", "H1", "frac_sandlike_0_10", "median_Ic_0_10", "n_ic_transitions_0_10", "PGAsig"]]
    D = D.merge(c11, on="rec_index")
    D["block"] = (np.floor(D["E"] / 1000).astype(int) * 100000 + np.floor(D["N"] / 1000).astype(int))
    D["env"] = D["gm3_environ"].fillna("unmapped")
    lf_counts = D["gm3_landform"].value_counts()
    keep_lf = lf_counts[lf_counts >= 40].index
    D["lf"] = np.where(D["gm3_landform"].isin(keep_lf), D["gm3_landform"], "other/unmapped")
    res["site_term_by_environ"] = D.groupby("env")["d_mean"].agg(["size", "mean", "median", "std"]).round(3).to_dict(orient="index")
    res["site_term_by_landform"] = D.groupby("lf")["d_mean"].agg(["size", "mean", "median"]).round(3).sort_values("mean").to_dict(orient="index")
    for label, f in [("env_only", "d_mean ~ C(env)"), ("landform_only", "d_mean ~ C(lf)"),
                     ("landform_plus_cpt", "d_mean ~ C(lf) + GWT + H1 + frac_sandlike_0_10 + median_Ic_0_10 + n_ic_transitions_0_10 + PGAsig")]:
        m = smf.ols(f, data=D).fit()
        res[f"R2_{label}"] = float(m.rsquared)
        res[f"R2adj_{label}"] = float(m.rsquared_adj)
    if "dist_channel_m" in D:
        D["dist_bin"] = pd.cut(D["dist_channel_m"], [0, 50, 100, 200, 400, 800, 1600, 1e6])
        res["site_term_by_channel_distance"] = D.groupby("dist_bin", observed=True)["d_mean"].agg(["size", "mean", "median"]).round(3).astype(float).to_dict(orient="index")
        res["site_term_by_channel_distance"] = {str(k): v for k, v in res["site_term_by_channel_distance"].items()}
        m = smf.ols("d_mean ~ np.log1p(dist_channel_m) + C(lf)", data=D).fit(cov_type="cluster", cov_kwds={"groups": D["block"]})
        res["log_dist_channel_coef_given_landform"] = [float(m.params["np.log1p(dist_channel_m)"]), float(m.bse["np.log1p(dist_channel_m)"]),
                                                       float(m.pvalues["np.log1p(dist_channel_m)"])]
    (out / "geomorph_site_terms.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:9000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
