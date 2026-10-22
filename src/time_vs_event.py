"""Elapsed time versus intervening earthquakes for the change in shallow sand resistance.

1. Archive: within-window slopes of the liquefiable-zone strata (ln qc of saturated sand-like
   layers, 1-5 m, 5-10 m, 1-10 m; deep sands 12-20 m) per decade of days since the window's opening
   earthquake, with 250-m cell fixed effects; and the implied change across each window.
   For comparison, the slope implied by Yang et al. (2026): a 69-72 % rise in sand-dominated
   qc1Ncs between t ~ 10 and 800 days after shaking, i.e. ln(1.70)/log10(80) = 0.28 per decade.
2. Repeat panel: consecutive increments within each family (orig -> a -> b -> d -> e), classified as
   spanning an earthquake or not, with cumulative change at each visit.

Usage: python -I time_vs_event.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery_tests import EV, fe_ols, load  # noqa: E402

YANG_SLOPE = float(np.log(1.70) / np.log10(800 / 10))


def archive(out):
    D = load(out).merge(pd.read_parquet(out / "negative_controls_strata.parquet"), on="rec_index", how="left")
    windows = {"P0_since_Darfield": ("2010", EV["2011a"], ["STAGE"]),
               "P2_since_13Jun2011": ("2011b", EV["2011c"], ["STAGE", "OTHER"]),
               "P3_since_23Dec2011": ("2011c", EV["2016"], ["POD", "TCR", "OTHER"]),
               "P3_POD_only": ("2011c", EV["2016"], ["POD"])}
    res = {"yang_implied_slope_per_decade": YANG_SLOPE}
    for name, (ev, end, progs) in windows.items():
        w = D[(D["date"] > EV[ev]) & (D["date"] < end) & D["programme"].isin(progs)].copy()
        days = (w["date"] - EV[ev]).dt.days.clip(lower=1)
        w["log10_days"] = np.log10(days)
        res[name] = {"n": int(len(w)), "days_p10_p90": [int(days.quantile(0.1)), int(days.quantile(0.9))],
                     "log10_span_p10_p90": float(np.log10(days.quantile(0.9) / days.quantile(0.1)))}
        for c in ["lnqc_sandW_1_5", "lnqc_sandW_5_10", "lnqc_sandW_1_10", "lnqc_sand_12_20"]:
            r = fe_ols(w, c, ["log10_days"])
            if r:
                b, se = r["coef"]["log10_days"], r["se"]["log10_days"]
                res[name][c] = {"slope": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "n_used": r["n"], "cells": r["n_fe_groups"],
                                "implied_change_p10_p90": b * res[name]["log10_span_p10_p90"],
                                "yang_implied_change_p10_p90": YANG_SLOPE * res[name]["log10_span_p10_p90"]}
    return res


def panel(out):
    P = pd.read_csv(out / "repeat_pairs.csv", parse_dates=["t0", "t1"])
    M = pd.read_csv(out / "repeat_members.csv", parse_dates=["date"])
    order = {"orig": 0, "a": 1, "b": 2, "c": 2.5, "d": 3, "extra": 3.5, "e": 4}
    inc = []
    for fam, g in M.groupby("fam"):
        seq = g.sort_values(["date", "suffix"], key=lambda s: s.map(order) if s.name == "suffix" else s)
        seq = seq[~seq["suffix"].isin(["c", "extra"])]  # same-day duplicates kept for the noise baseline only
        sfx = seq["suffix"].tolist()
        cum = 0.0
        inc.append({"fam": fam, "from": None, "to": sfx[0], "date": seq["date"].iloc[0], "cum_dlogq_1_10": 0.0, "cum_dq_1_10": 0.0})
        cumq = 0.0
        for a, b in zip(sfx[:-1], sfx[1:]):
            r = P[(P["fam"] == fam) & (((P["from"] == a) & (P["to"] == b)) | ((P["from"] == b) & (P["to"] == a)))].iloc[0]
            sign = 1.0 if r["from"] == a else -1.0
            cum += sign * r["dlogq_1_10"]
            cumq += sign * r["dq_1_10"]
            inc.append({"fam": fam, "from": a, "to": b, "date": r["t1"] if sign > 0 else r["t0"], "days": int(r["days"]),
                        "class": r["class"], "n_events": int(r["n_events"]), "sep_m": float(r["sep_m"]),
                        "dlogq_1_10": sign * r["dlogq_1_10"], "dlogq_1_4": sign * r["dlogq_1_4"], "dlogq_4_8": sign * r["dlogq_4_8"],
                        "dlogq_8_12": sign * r["dlogq_8_12"], "cum_dlogq_1_10": cum, "cum_dq_1_10": cumq})
    I = pd.DataFrame(inc)
    I.to_csv(out / "repeat_increments.csv", index=False)
    J = I[I["from"].notna()]
    summ = {}
    for key, g in J.groupby(J["n_events"] > 0):
        summ["spanning" if key else "event_free"] = {"n_increments": int(len(g)), "sum_dlogq_1_10": float(g["dlogq_1_10"].sum()),
                                                     "median": float(g["dlogq_1_10"].median()),
                                                     "days_total": int(g["days"].sum()),
                                                     "per_increment": g[["fam", "from", "to", "days", "dlogq_1_10"]].round(3).values.tolist()}
    return summ, I


def main(out_dir):
    out = Path(out_dir)
    res = {"archive": archive(out)}
    res["panel"], I = panel(out)
    (out / "time_vs_event.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))
    print(I.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
