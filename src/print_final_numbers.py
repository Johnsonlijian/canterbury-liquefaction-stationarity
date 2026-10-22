"""Print a compact view of final_numbers.json (for manuscript checking).

Usage: python -I print_final_numbers.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd


def fmt(v):
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, list):
        return [fmt(x) for x in v]
    return v


def main(out_dir):
    f = json.loads((Path(out_dir) / "final_numbers.json").read_text())
    for k, v in f["paperA"].items():
        print(k, {kk: fmt(vv) for kk, vv in v.items()})
    T = pd.DataFrame(f["paperB"]["auc_by_event_period"])
    for ix in ["LSN", "LPI", "LPIish"]:
        T[ix] = T.apply(lambda r: f"{r['AUC_' + ix]:.2f} [{r['AUC_' + ix + '_ci95'][0]:.2f}, {r['AUC_' + ix + '_ci95'][1]:.2f}]", axis=1)
    print(T[["event", "period", "n", "n_pos", "base", "LSN", "LPI", "LPIish"]].to_string())
    for ix, d in f["paperB"]["did_2010_by_feb2011_label"].items():
        print(ix, {k: fmt(v) for k, v in d.items()})
    print(json.dumps(f["paperB"]["pooled_vs_within"], indent=0))


if __name__ == "__main__":
    main(sys.argv[1])
