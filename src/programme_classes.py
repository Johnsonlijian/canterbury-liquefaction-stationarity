"""Classify PRJ-2937 soundings into investigation programmes from NZGD naming conventions.

Patterns (case-insensitive) applied to the NZGD investigation name, falling back to the
PRJ-2937 CPT name:
  STAGE   CPT-XXX-nn[a-z]   suburb-coded area investigation (e.g. CPT-BUR-22)
  POD     XXX-PODnn-CPTnnn  POD-coded programme (e.g. NNB-POD06-CPT23)
  TCR     XXX-TCRnn-CPTnnn  TCR-coded programme
  OTHER   anything else
Writes outputs/programme_classes.parquet and a cross-tabulation by test period.

Usage: python -I programme_classes.py <outputs_dir>
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

PATTERNS = [("STAGE", re.compile(r"^CPT-[A-Z]{2,4}-\d+[A-Z]?", re.I)),
            ("POD", re.compile(r"^[A-Z]{2,4}-POD\d+-CPT\d+", re.I)),
            ("TCR", re.compile(r"^[A-Z]{2,4}-TCR\d+-CPT\d+", re.I))]


def classify(name):
    s = str(name).strip()
    for lab, pat in PATTERNS:
        if pat.search(s):
            return lab
    return "OTHER"


def main(out_dir):
    out = Path(out_dir)
    snd = pd.read_parquet(out / "soundings.parquet")
    link = pd.read_parquet(out / "nzgd_link.parquet")
    S = snd.merge(link, on="rec_index", how="left")
    S["name_for_class"] = S["nzgd_name"].fillna(S["CPTname"])
    S["programme"] = S["name_for_class"].map(classify)
    # suburb code where present
    S["suburb_code"] = S["name_for_class"].astype(str).str.extract(r"^(?:CPT-)?([A-Z]{2,4})-", flags=re.I)[0].str.upper()
    S["period"] = pd.cut(S["date"], [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-02-22"), pd.Timestamp("2011-06-13"),
                                    pd.Timestamp("2011-12-23"), pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")],
                         right=False, labels=["P0", "P1", "P2", "P3", "P4"])
    S[["rec_index", "programme", "suburb_code", "nzgd_name", "ground_improvement", "ground_level", "period"]].to_parquet(
        out / "programme_classes.parquet", index=False)
    ct = pd.crosstab(S["programme"], S["period"])
    res = {"crosstab_programme_by_period": {str(k): v for k, v in ct.to_dict().items()},
           "examples": {p: S.loc[S["programme"] == p, "name_for_class"].head(6).tolist() for p in S["programme"].unique()},
           "ground_improvement_by_period": S.groupby("period", observed=True)["ground_improvement"].apply(lambda x: int(x.fillna(False).sum())).to_dict(),
           "median_date_by_programme": S.groupby("programme")["date"].median().astype(str).to_dict()}
    (out / "programme_classes.json").write_text(json.dumps(res, indent=2, default=str))
    print(ct)
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
