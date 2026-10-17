"""Staged simulation following the thesis plan: 2 levels -> 3 levels -> 4 levels.

All stages use a design shaped like the ARCOS data (7 regions, 67 counties with
unequal numbers per region, 1,483 ZIP codes with unequal numbers per county,
T = 60 months) and the exact maximum-likelihood fits of real_design_study.py.

Stage 1 (2 levels: months within ZIP). Candidates: no random effect, or ZIP.
         This is the setting where a BIC has already been derived (Cho et al., 2024).
Stage 2 (3 levels: months within ZIP within county). Candidates: all subsets of {ZIP, county}.
Stage 3 (4 levels: ... within region). Candidates: all subsets of {ZIP, county, region}.

Methods: AIC; standard BIC (k log N, R/lme4 default); BIC with the number of
top-level clusters (SAS-style); backward boundary-corrected likelihood ratio
tests at 5%; the new exact-information BIC; and, in stages 2 and 3, "2-level
analysis only": the exact-information BIC restricted to the 2-level candidates
(no random effect or ZIP), i.e. what happens if county and region are ignored.
"""
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_design_study as rd  # noqa: E402

T = 60
REPS = int(os.environ.get("REPS", 400))
LEVELS = rd.LEVELS


def subsets(levels):
    return [frozenset(s) for r in range(len(levels) + 1) for s in itertools.combinations(levels, r)]


STAGES = {
    "Stage 1: 2 levels": (("ZIP",), [
        ("No ZIP effect",          {"ZIP": 0.0,  "county": 0.0, "region": 0.0}),
        ("Small ZIP effect",       {"ZIP": 0.004, "county": 0.0, "region": 0.0}),
        ("Medium ZIP effect",      {"ZIP": 0.5,  "county": 0.0, "region": 0.0}),
    ]),
    "Stage 2: 3 levels": (("ZIP", "county"), [
        ("ZIP only (no county)",   {"ZIP": 0.5, "county": 0.0,  "region": 0.0}),
        ("Small county effect",    {"ZIP": 0.5, "county": 0.008, "region": 0.0}),
        ("Medium county effect",   {"ZIP": 0.5, "county": 0.3,  "region": 0.0}),
    ]),
    "Stage 3: 4 levels": (("ZIP", "county", "region"), [
        ("No region effect",       {"ZIP": 0.5, "county": 0.3, "region": 0.0}),
        ("Small region effect",    {"ZIP": 0.5, "county": 0.3, "region": 0.05}),
        ("Medium region effect",   {"ZIP": 0.5, "county": 0.3, "region": 0.3}),
        ("Large region effect",    {"ZIP": 0.5, "county": 0.3, "region": 1.0}),
        ("Region but no county",   {"ZIP": 0.5, "county": 0.0, "region": 0.3}),
    ]),
}


def choose(design, stats, levels):
    models = subsets(levels)
    fits = {m: rd.fit(design, stats, m) for m in models}
    units = {"e": design.N, "ZIP": design.J, "county": design.K, "region": design.R}

    def best(score, among=models):
        return min(among, key=lambda m: score[m])

    out = {}
    k = {m: 2 + len(m) for m in models}
    top = {m: max((l for l in LEVELS if l in m), key=LEVELS.index, default="e") for m in models}
    m2ll = {m: -2 * fits[m][0] for m in models}
    out["AIC"] = best({m: m2ll[m] + 2 * k[m] for m in models})
    out["Standard BIC (N)"] = best({m: m2ll[m] + k[m] * np.log(units["e"]) for m in models})
    out["BIC (top clusters)"] = best({m: m2ll[m] + k[m] * np.log(units[top[m]]) for m in models})
    exact = {m: m2ll[m] + rd.log_det_expected_info(design, fits[m][1], m) for m in models}
    out["New BIC (exact information)"] = best(exact)
    cur = frozenset(levels)
    while cur:
        p = {}
        for l in cur:
            lr = max(0.0, m2ll[cur - {l}] - m2ll[cur])
            p[l] = 0.5 * chi2.sf(lr, 1) if lr > 0 else 1.0
        worst = max(p, key=p.get)
        if p[worst] <= 0.05:
            break
        cur = cur - {worst}
    out["Likelihood ratio tests"] = cur
    if len(levels) > 1:
        out["2-level analysis only"] = best(exact, among=[frozenset(), frozenset({"ZIP"})])
    return out


_design = None


def _init():
    global _design
    rc, cz = rd.make_design()
    _design = rd.Design(rc, cz, T)


def one_rep(args):
    stage, s_idx, seed = args
    levels, scenarios = STAGES[stage]
    var = dict(scenarios[s_idx][1], e=1.0)
    rng = np.random.default_rng(seed)
    W, m = rd.simulate(_design, var, rng)
    return stage, s_idx, choose(_design, rd.suff(_design, W, m), levels)


if __name__ == "__main__":
    rows = []
    with Pool(initializer=_init) as pool:
        for stage, (levels, scenarios) in STAGES.items():
            for s_idx, (label, var) in enumerate(scenarios):
                t0 = time.time()
                jobs = [(stage, s_idx, hash((stage, s_idx)) % 10**6 * 1000 + r) for r in range(REPS)]
                res = [c for _, _, c in pool.imap_unordered(one_rep, jobs, chunksize=8)]
                truth = frozenset(l for l in levels if var[l] > 0)
                for meth in res[0]:
                    picks = [c[meth] for c in res]
                    rows.append({
                        "stage": stage, "scenario": label, "true_structure": rd.name(truth),
                        "method": meth,
                        "pct_correct": round(100 * np.mean([p == truth for p in picks]), 1),
                        "pct_added_false_level": round(100 * np.mean([bool(p - truth) for p in picks]), 1),
                        "pct_missed_real_level": round(100 * np.mean([bool(truth - p) for p in picks]), 1)})
                print(f"{stage} | {label:24s} | {time.time() - t0:5.1f}s |",
                      {r["method"]: r["pct_correct"] for r in rows
                       if r["stage"] == stage and r["scenario"] == label}, flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/staged_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
