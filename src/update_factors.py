"""Practical update factors: how observed manifestation history shifts a site's odds.

Uses the empirical-Bayes site terms fitted on 2010 + 2011 only (training events) to report,
for each 2010/2011 outcome history, the mean posterior site term, the implied odds
multiplier exp(lambda * d) for lambda = 1 and the 2016-calibrated shrinkage lambda = 0.5,
and the observed 2016 manifestation rate by history (out-of-sample check).

Usage: python -I update_factors.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_terms_glmm import fit, eb_from_params  # noqa: E402

FEB11 = pd.Timestamp("2011-02-22")


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_parquet(out / "cases.parquet")
    c = c[(c["variant"] == "inv") & c["y"].notna()].copy()
    c["p0"] = (c["date"] < FEB11).astype(int)
    res = {}
    for ix in ["LSN", "LPI", "LPIish"]:
        c["x"] = np.log1p(c[ix])
        o, eb, theta, names = fit(c, ["2010", "2011"], epoch_split_2010=True)
        ebh = eb_from_params(c[c["event"].isin(["2010", "2011"])], theta, names)
        w = c.pivot_table(index="rec_index", columns="event", values="y")
        w.columns = [f"y{e}" for e in w.columns]
        D = ebh.merge(w, left_on="rec_index", right_index=True)
        D = D.dropna(subset=["y2010", "y2011"])
        D["history"] = np.select([(D["y2010"] == 1) & (D["y2011"] == 1), (D["y2010"] == 0) & (D["y2011"] == 1),
                                  (D["y2010"] == 1) & (D["y2011"] == 0)],
                                 ["2010 and 2011", "2011 only", "2010 only"], default="neither")
        tab = {}
        for h, g in D.groupby("history"):
            tab[h] = {"n": int(len(g)), "mean_site_term": float(g["d_hat"].mean()),
                      "odds_factor_lambda1": float(np.exp(g["d_hat"].mean())),
                      "odds_factor_lambda0.5": float(np.exp(0.5 * g["d_hat"].mean())),
                      "observed_2016_rate": float(g["y2016"].mean()) if g["y2016"].notna().any() else None,
                      "n_2016": int(g["y2016"].notna().sum())}
        res[ix] = {"tau_train_2010_2011": o["tau"], "by_history": tab}
    (out / "update_factors.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
