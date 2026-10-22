"""Generate the Paper B provenance numbers (skill by test period, footprint, truncation, label
resolution, spatial composition) as LaTeX macros (paperB/numbers_prov.tex).

Composition: the northern area is latitude > -43.42 (Kaiapoi/Waimakariri); shares are of dated
soundings in the case table (ground-improved soundings excluded upstream).

Usage: python -I paperB_prov_numbers.py <outputs_dir> <paperB_dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINUS = re.compile(r"(?<![\w$-])-(?=\d)")
IX = ["LSN", "LPI", "LPIish"]


def j(p):
    return json.loads(Path(p).read_text())


def rng(vals, fmt="{:.2f}"):
    a, b = (fmt.format(v).replace("-0.00", "0.00") for v in (min(vals), max(vals)))
    return a if a == b else f"{a}--{b}"


def main(out_dir, paper_dir):
    o, P = Path(out_dir), Path(paper_dir)
    fn = j(o / "final_numbers.json")["paperB"]
    tm, tp, tl, ex = j(o / "timing_matched.json"), j(o / "timing_protocol_check.json"), j(o / "timing_label_checks.json"), j(o / "explore_sequence.json")
    N = {}
    T = pd.DataFrame(fn["auc_by_event_period"])
    def aucs(ev, per):
        r = T[(T["event"] == ev) & (T["period"] == per)].iloc[0]
        return [r[f"AUC_{ix}"] for ix in IX], r
    a0, r0 = aucs("2010", "P0")
    a3, r3 = aucs("2010", "P3")
    N["PZERORANGE"] = rng(a0)
    N["PTHREERANGE"] = rng(a3)
    N["PZEROMAX"] = f"{max(a0):.2f}"
    N["PTHREEMID"] = f"{np.median(a3):.2f}"
    N["NPZEROTEN"] = f"{int(r0['n']):,}"
    a1, _ = aucs("2010", "P1")
    a2, _ = aucs("2010", "P2")
    N["PONERANGE"] = rng(a1)
    N["PTWORANGE"] = rng(a2)
    pe = {r["event"]: r for r in ex["per_event_auc"] if r["variant"] == "inv"}
    N["ALLTENRANGE"] = rng([pe["2010"][f"AUC_{ix}"] for ix in IX])
    # footprint (inverse profiles)
    fp10 = {f["radius_m"]: f for f in tm["2010_inv"]["footprint"]}
    for ix, nm in zip(IX, ["LSN", "LPI", "ISHX"]):
        f = fp10[100]
        N[f"FPHUNDRED{nm}"] = f"{f[f'dAUC_{ix}_early_minus_late']:.2f} ({f[f'dAUC_{ix}_ci95'][0]:.2f}--{f[f'dAUC_{ix}_ci95'][1]:.2f})"
    N["FPRANGEALL"] = rng([fp10[r][f"dAUC_{ix}_early_minus_late"] for r in fp10 for ix in IX])
    N["FPWIDERANGE"] = rng([fp10[r][f"dAUC_{ix}_early_minus_late"] for r in fp10 if r >= 100 for ix in IX])
    f50 = fp10[50]
    N["FPFIFTYRANGE"] = rng([f50[f"dAUC_{ix}_early_minus_late"] for ix in IX])
    N["FPFIFTYNLATE"] = f"{int(f50['n_late']):,}"
    N["FPFIFTYZERO"] = "include" if all(f50[f"dAUC_{ix}_ci95"][0] <= 0 for ix in IX) else ("some include" if any(f50[f"dAUC_{ix}_ci95"][0] <= 0 for ix in IX) else "exclude")
    fp11 = tm["2011_inv"]["footprint"]
    N["FPELEVENRANGE"] = rng([f[f"dAUC_{ix}_early_minus_late"] for f in fp11 for ix in IX])
    N["FPELEVENINCLZERO"] = "all" if all(f[f"dAUC_{ix}_ci95"][0] <= 0 <= f[f"dAUC_{ix}_ci95"][1] for f in fp11 for ix in IX) else "not all"
    wc = tm["2010_inv"]["within_cell500"]
    N["WCEARLY"] = rng([wc[f"withincell_AUC_{ix}_early"] for ix in IX])
    N["WCLATE"] = rng([wc[f"withincell_AUC_{ix}_late"] for ix in IX])
    # truncation (LSN etc. at z0 = 1, 1.5, 2 m, all footprint)
    tr = pd.DataFrame(tp["truncation_test_2010"])
    tr = tr[(tr["footprint"] == "all") & tr["z0"].isin([1.0, 1.5, 2.0])]
    tr["per"] = tr["period"].str[:2]
    N["TRUNCPZERO"] = rng(tr.loc[tr["per"] == "P0", [f"AUC_{ix}" for ix in IX]].values.ravel())
    N["TRUNCPTHREE"] = rng(tr.loc[tr["per"] == "P3", [f"AUC_{ix}" for ix in IX]].values.ravel())
    # label resolution within 50 m
    l3 = pd.DataFrame(tl["L3_neighbour_label_agreement"])
    l3 = l3[l3["radius_m"] == 50].set_index("period")
    for per, nm in zip(["P0", "P1", "P2", "P3"], ["ZERO", "ONE", "TWO", "THREE"]):
        N[f"AGREE{nm}"] = f"{100 * l3.loc[per, 'agree2010_exact']:.0f}"
    N["AGREEELEVEN"] = rng([100 * l3.loc[p, "agree2011_binary"] for p in ["P0", "P1", "P2", "P3"]], "{:.0f}")
    # severe vs none (2010)
    l1 = pd.DataFrame(tl["L1_severe_vs_none"])
    l1 = l1[(l1["event"] == "2010") & (l1["footprint"] == "all")].set_index("period")
    N["SEVEREPZERO"] = rng([l1.loc["P0", f"AUC_{ix}"] for ix in IX])
    N["SEVEREPTHREE"] = rng([l1.loc["P3", f"AUC_{ix}"] for ix in IX])
    # composition
    C = pd.read_parquet(o / "cases.parquet")
    S = pd.read_parquet(o / "soundings.parquet")[["rec_index", "lat"]]
    pc = pd.read_parquet(o / "programme_classes.parquet")[["rec_index", "period"]]
    s = C[(C["variant"] == "inv") & (C["event"] == "2010")][["rec_index", "y"]].merge(S, on="rec_index").merge(pc, on="rec_index")
    s = s[s["period"].notna()]
    early = s["period"] == "P0"
    north = s["lat"] > -43.42
    N["NORTHEARLY"] = f"{100 * north[early].mean():.0f}"
    N["NORTHLATE"] = f"{100 * north[~early].mean():.0f}"
    N["MANTENEARLY"] = f"{100 * s.loc[early, 'y'].mean():.0f}"
    N["MANTENPTHREE"] = f"{100 * s.loc[s['period'] == 'P3', 'y'].mean():.0f}"
    lines = ["\\newcommand{\\" + k + "}{" + MINUS.sub("$-$", v) + "}" for k, v in N.items()]
    (P / "numbers_prov.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (o / "paperB_prov_numbers.json").write_text(json.dumps(N, indent=1))
    for k, v in N.items():
        print(f"{k:<16} {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
