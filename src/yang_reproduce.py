"""Which per-sounding definition reproduces the period statistics reported by Yang et al. (2026)?

Targets read from Yang et al. (2026), Section 3.2 and Fig. 5-6:
  groups  G1 4 Sep 2010-22 Feb 2011 (~440 soundings), G2 23 Feb-31 Dec 2011 (~650),
          G3 2012 (~2550), G4 2013-2014 (~1000)
  qc1Ncs of liquefiable layers: G1 median 58 (IQR 45-70); means G2 ~66, G3 ~72, G4 ~76
  liquefiable-layer depth: G1 5.39 +/- 3.11, G2 6.76 +/- 3.92, G3 6.70 +/- 3.97, 2013 5.70 +/- 3.09 m
  thickness (fitted mean): ~0.70 m in 2010-2011 to 0.495 m in 2013
  depth bands (2010-2011 -> 2013-2014): 0-5 m 64 -> 83; 5-10 m 76 -> 87; >10 m 78 -> 107
Exclusions as stated: pre-drill depth unknown or > 2.5 m; unknown manifestation (tested variants).

Usage: python -I yang_reproduce.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = {"G1_median": 58.0, "G1_q25": 45.0, "G1_q75": 70.0, "G2_mean": 66.0, "G3_mean": 72.0, "G4_mean": 76.0}
TARGET_Z = {"G1": (5.39, 3.11), "G2": (6.76, 3.92), "G3": (6.70, 3.97), "G4": (5.70, 3.09)}


def groups(date):
    return pd.cut(date, [pd.Timestamp("2010-09-04"), pd.Timestamp("2011-02-23"), pd.Timestamp("2012-01-01"),
                         pd.Timestamp("2013-01-01"), pd.Timestamp("2015-01-01")], right=False,
                  labels=["G1", "G2", "G3", "G4"])


def load(out):
    M = pd.read_parquet(out / "yang_metrics.parquet")
    S = pd.read_parquet(out / "soundings.parquet")[["rec_index", "date", "pd", "lat", "lon"]]
    C = pd.read_parquet(out / "cases.parquet")
    lab = C[C["variant"] == "meas"].pivot(index="rec_index", columns="event", values="label")
    lab.columns = [f"label_{c}" for c in lab.columns]
    P = pd.read_parquet(out / "programme_classes.parquet")[["rec_index", "programme", "ground_improvement", "period"]]
    D = S.merge(M, on="rec_index").merge(lab.reset_index(), on="rec_index").merge(P, on="rec_index")
    D["G"] = groups(D["date"])
    return D


def summarise(D, col):
    r = {}
    for g, x in D.groupby("G", observed=True)[col]:
        x = x.dropna()
        r[g] = {"n": int(x.size), "median": float(x.median()) if x.size else np.nan,
                "q25": float(x.quantile(0.25)) if x.size else np.nan, "q75": float(x.quantile(0.75)) if x.size else np.nan,
                "mean": float(x.mean()) if x.size else np.nan, "sd": float(x.std()) if x.size else np.nan}
    return r


def score(s):
    try:
        e = [(s["G1"]["median"] - TARGET["G1_median"]), (s["G1"]["q25"] - TARGET["G1_q25"]), (s["G1"]["q75"] - TARGET["G1_q75"]),
             (s["G2"]["mean"] - TARGET["G2_mean"]), (s["G3"]["mean"] - TARGET["G3_mean"]), (s["G4"]["mean"] - TARGET["G4_mean"])]
        return float(np.sqrt(np.mean(np.square(e))))
    except KeyError:
        return np.nan


def main(out_dir):
    out = Path(out_dir)
    D = load(out)
    filters = {
        "pd_ok": D["pd"].notna() & (D["pd"] <= 2.5),
        "pd_ok_lab2011": D["pd"].notna() & (D["pd"] <= 2.5) & D["label_2011"].between(0, 5),
        "pd_ok_laball": D["pd"].notna() & (D["pd"] <= 2.5) & D[["label_2010", "label_2011", "label_2016"]].apply(lambda c: c.between(0, 5)).all(axis=1),
        "pd_ok_labany": D["pd"].notna() & (D["pd"] <= 2.5) & D[["label_2010", "label_2011", "label_2016"]].apply(lambda c: c.between(0, 5)).any(axis=1),
    }
    res = {"counts": {k: D[f].groupby(D.loc[f, "G"], observed=True).size().to_dict() for k, f in filters.items()}}
    # Dprev: choose the demand of the most recent event before the test date
    for proc in ["IB08", "BI14"]:
        for var in ["meas", "inv"]:
            for kind in ["crit_q", "layer_q", "all_q", "crit_z", "layer_z", "all_z", "layer_thick", "all_thick", "layer_ic", "crit_ic"]:
                c10, c11, c16 = (f"{kind}_{proc}_D{e}_{var}" for e in ("2010", "2011", "2016"))
                if c10 in D:
                    prev = np.where(D["date"] < pd.Timestamp("2011-02-22"), D[c10],
                                    np.where(D["date"] < pd.Timestamp("2016-02-14"), D[c11], D[c16]))
                    D[f"{kind}_{proc}_Dprev_{var}"] = prev
    rows = []
    F = filters["pd_ok_lab2011"]
    for proc in ["IB08", "BI14"]:
        for var in ["meas", "inv"]:
            for dem in ["D2010", "D2011", "Dprev"]:
                for kind in ["crit_q", "layer_q", "all_q"]:
                    col = f"{kind}_{proc}_{dem}_{var}"
                    s = summarise(D[F], col)
                    zkind = kind.replace("_q", "_z")
                    sz = summarise(D[F], f"{zkind}_{proc}_{dem}_{var}")
                    tk = "layer_thick" if kind != "all_q" else "all_thick"
                    st = summarise(D[F], f"{tk}_{proc}_{dem}_{var}") if kind != "crit_q" else {}
                    rows.append({"proc": proc, "var": var, "demand": dem, "kind": kind, "rmse_vs_yang": score(s),
                                 **{f"{g}_{k}": round(v[k], 1) for g, v in s.items() for k in ("n", "median", "q25", "q75", "mean")},
                                 **{f"{g}_z": f"{v['mean']:.2f}+/-{v['sd']:.2f}" for g, v in sz.items()},
                                 **{f"{g}_thick_mean": round(v["mean"], 2) for g, v in st.items()},
                                 **{f"{g}_thick_median": round(v["median"], 2) for g, v in st.items()}})
    R = pd.DataFrame(rows).sort_values("rmse_vs_yang")
    R.to_csv(out / "yang_reproduce_candidates.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(json.dumps(res["counts"], indent=1, default=str))
    show = ["proc", "var", "demand", "kind", "rmse_vs_yang", "G1_n", "G1_median", "G1_q25", "G1_q75", "G2_mean", "G3_mean", "G4_mean"]
    print(R[show].head(12).to_string(index=False))
    print(R[["proc", "var", "demand", "kind", "G1_z", "G2_z", "G3_z", "G4_z", "G1_thick_mean", "G4_thick_mean", "G1_thick_median", "G4_thick_median"]].head(12).to_string(index=False))
    (out / "yang_reproduce_counts.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
