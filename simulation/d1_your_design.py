"""Direction 1 pilot using a design close to the real ARCOS data.

Real data: 7 regions, 67 counties, 1,483 ZIP codes. Approximated by a balanced
design of 7 regions x 10 counties x 21 ZIP codes (70 counties, 1,470 ZIPs) and
60 months. Question: with only 7 regions, how often does each criterion
correctly decide whether region-level differences exist?

Scenarios: no region effect (truth = ZIP + county), and a region effect that is
small, medium or large relative to the county effect.
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_depth_4level import MODELS, LEVELS, fit, strata, log_det_info, fixed_penalties  # noqa: E402

R, K, Z, T, REPS = 7, 10, 21, 60, 500
VAR = {"ZIP": 0.5, "county": 0.3}
SCENARIOS = [("No region effect", 0.0),
             ("Small region effect", 0.05),
             ("Medium region effect", 0.3),
             ("Large region effect", 1.0)]
CRITS = ["AIC", "BIC (N)", "BIC (top clusters)", "BIC (level-specific)", "BIC (exact information)"]


def run(s2R, rng):
    truth = frozenset({"ZIP", "county", "region"} if s2R > 0 else {"ZIP", "county"})
    fp = {m: fixed_penalties(m, R, K, Z, T) for m in MODELS}
    hits = dict.fromkeys(CRITS, 0)
    for _ in range(REPS):
        y = (rng.normal(0, 1, (R, K, Z, T))
             + rng.normal(0, np.sqrt(VAR["ZIP"]), (R, K, Z, 1))
             + rng.normal(0, np.sqrt(VAR["county"]), (R, K, 1, 1))
             + rng.normal(0, np.sqrt(s2R), (R, 1, 1, 1)))
        S, d = strata(y)
        fits = {m: fit(S, d, m) for m in MODELS}
        for c in CRITS:
            if c == "BIC (exact information)":
                score = {m: -2 * ll + log_det_info(m, lam, d, K, Z, T) for m, (ll, lam) in fits.items()}
            else:
                score = {m: -2 * fits[m][0] + fp[m][c] for m in MODELS}
            hits[c] += min(score, key=score.get) == truth
    return {c: round(100 * h / REPS, 1) for c, h in hits.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(67)
    rows = []
    for label, s2R in SCENARIOS:
        res = run(s2R, rng)
        print(f"{label:22s}", res)
        for c, v in res.items():
            rows.append({"scenario": label, "region_variance": s2R, "criterion": c, "pct_correct": v})
    os.makedirs("results", exist_ok=True)
    with open("results/d1_your_design.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
