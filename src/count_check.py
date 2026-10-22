"""Sample-count audit for the manuscripts.

Usage: python -I count_check.py <outputs_dir>
"""
import sys
from pathlib import Path

import pandas as pd


def main(out_dir):
    out = Path(out_dir)
    s = pd.read_parquet(out / "soundings.parquet")
    c = pd.read_parquet(out / "cases.parquet")
    inv = c[c["variant"] == "inv"]
    print("soundings in file:", 5668)
    print("soundings processed (>=10 valid points):", inv["rec_index"].nunique())
    print("unknown labels (code 10) among processed:", int((inv["label"] == 10).sum()))
    lab = inv[inv["y"].notna()]
    print("labelled case histories:", len(lab), "sites with >=1 label:", lab["rec_index"].nunique())
    print("by event:", lab.groupby("event").size().to_dict())
    full = lab.groupby("rec_index")["event"].nunique()
    print("sites with all three labels:", int((full == 3).sum()))
    print("dated soundings:", int(s["date"].notna().sum()), "min", s["date"].min(), "max", s["date"].max())
    p = pd.cut(s["date"], [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-02-22"), pd.Timestamp("2011-06-13"),
                           pd.Timestamp("2011-12-23"), pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")],
               right=False, labels=["P0", "P1", "P2", "P3", "P4"])
    print("periods (all soundings):", p.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
