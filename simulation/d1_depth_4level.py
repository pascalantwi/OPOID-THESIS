"""Pilot simulation for Direction 1 with up to four levels.

Balanced design: T months within Z ZIP codes within K counties within R regions,

    y = mu + r_region + c_county + z_zip + e,
    variances s2_R, s2_C, s2_Z, s2_e.

The true structure has depth 2 (ZIP only), 3 (ZIP + county), 4 (ZIP + county +
region), or 4 with the middle level missing (ZIP + region). All 8 candidate
structures (every subset of {ZIP, county, region}) are fitted by exact maximum
likelihood and the structure is selected by AIC, BIC with total N, BIC with the
number of top-level clusters, the leading-order level-specific BIC and the
exact-information BIC.

The balanced covariance has four strata with eigenvalues
    lam1 = s2_e,  lam2 = lam1 + T s2_Z,  lam3 = lam2 + Z T s2_C,  lam4 = lam3 + K Z T s2_R;
a level that is absent merges its stratum with the one below.
"""
import itertools
import numpy as np
from scipy.optimize import minimize

LOG2PI = np.log(2 * np.pi)
LEVELS = ("ZIP", "county", "region")          # random-effect levels, bottom to top


def strata(y):
    R, K, Z, T = y.shape
    m_z = y.mean(axis=3)                       # ZIP means      (R, K, Z)
    m_c = y.mean(axis=(2, 3))                  # county means   (R, K)
    m_r = y.mean(axis=(1, 2, 3))               # region means   (R,)
    m = y.mean()
    S = np.array([
        ((y - m_z[..., None]) ** 2).sum(),               # months within ZIP
        T * ((m_z - m_c[..., None]) ** 2).sum(),         # ZIPs within county
        Z * T * ((m_c - m_r[:, None]) ** 2).sum(),       # counties within region
        K * Z * T * ((m_r - m) ** 2).sum(),              # regions (mean profiled out)
    ])
    d = np.array([R * K * Z * (T - 1), R * K * (Z - 1), R * (K - 1), R])
    return S, d


def pava(s, w):
    blocks = [[si * wi, wi, 1] for si, wi in zip(s, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            a, b = blocks[i], blocks.pop(i + 1)
            blocks[i] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for tot, wt, n in blocks:
        out += [tot / wt] * n
    return np.array(out)


def groups_for(present):
    """Strata sharing an eigenvalue: stratum m+1 starts a new group only if level m is present."""
    groups = [[0]]
    for m, lev in enumerate(LEVELS, start=1):
        if lev in present:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


def fit(S, d, present):
    groups = groups_for(present)
    SS = np.array([S[g].sum() for g in groups])
    dd = np.array([d[g].sum() for g in groups])
    lam = pava(SS / dd, dd)
    ll = -0.5 * (d.sum() * LOG2PI + (dd * np.log(lam)).sum() + (SS / lam).sum())
    lam_strata = np.empty(4)
    for g, l in zip(groups, lam):
        lam_strata[g] = l
    return ll, lam_strata


def log_det_info(present, lam, d, K, Z, T):
    """Exact log det Fisher information (intercept + variance components) at the fit."""
    cols = {"e": [1, 1, 1, 1], "ZIP": [0, T, T, T],
            "county": [0, 0, Z * T, Z * T], "region": [0, 0, 0, K * Z * T]}
    A = np.array([cols[p] for p in ("e",) + tuple(l for l in LEVELS if l in present)],
                 dtype=float).T
    I_theta = A.T @ np.diag(d / (2 * lam ** 2)) @ A
    return np.log(d.sum() / lam[3]) + np.linalg.slogdet(I_theta)[1]


MODELS = [frozenset(s) for r in range(4) for s in itertools.combinations(LEVELS, r)]


def name(present):
    return "+".join(l for l in LEVELS if l in present) or "none"


def fixed_penalties(present, R, K, Z, T):
    units = {"e": R * K * Z * T, "ZIP": R * K * Z, "county": R * K, "region": R}
    k = 2 + len(present)                                   # intercept, s2_e, level variances
    top = max((l for l in LEVELS if l in present), key=LEVELS.index, default="e")
    level_specific = (np.log(units["e"]) + sum(np.log(units[l]) for l in present)
                      + np.log(units[top]))                # intercept pays its top level
    return {"AIC": 2 * k, "BIC (N)": k * np.log(units["e"]),
            "BIC (top clusters)": k * np.log(units[top]),
            "BIC (level-specific)": level_specific}


def check_ml(rng):
    R, K, Z, T = 4, 3, 3, 5
    for _ in range(15):
        y = (rng.normal(0, .5, (R, 1, 1, 1)) + rng.normal(0, .5, (R, K, 1, 1))
             + rng.normal(0, .7, (R, K, Z, 1)) + rng.normal(0, 1, (R, K, Z, T)))
        S, d = strata(y)

        def nll(p):
            se, sz, sc, sr = p
            lam = np.cumsum([se, T * sz, Z * T * sc, K * Z * T * sr])
            return 0.5 * (d.sum() * LOG2PI + (d * np.log(lam)).sum() + (S / lam).sum())

        best = min((minimize(nll, x0, method="L-BFGS-B",
                             bounds=[(1e-8, None)] + [(0, None)] * 3)
                    for x0 in itertools.product([0.5, 2], [0, .5], [0, .5], [0, .5])),
                   key=lambda r: r.fun)
        assert abs(-best.fun - fit(S, d, set(LEVELS))[0]) < 1e-5


def simulate(R, K, Z, T, var, truth, reps, rng):
    crits = ["AIC", "BIC (N)", "BIC (top clusters)", "BIC (level-specific)",
             "BIC (exact information)"]
    hits = dict.fromkeys(crits, 0)
    fp = {m: fixed_penalties(m, R, K, Z, T) for m in MODELS}
    for _ in range(reps):
        y = rng.normal(0, 1, (R, K, Z, T))
        if "ZIP" in truth:    y = y + rng.normal(0, np.sqrt(var["ZIP"]), (R, K, Z, 1))
        if "county" in truth: y = y + rng.normal(0, np.sqrt(var["county"]), (R, K, 1, 1))
        if "region" in truth: y = y + rng.normal(0, np.sqrt(var["region"]), (R, 1, 1, 1))
        S, d = strata(y)
        fits = {m: fit(S, d, m) for m in MODELS}
        for c in crits:
            if c == "BIC (exact information)":
                score = {m: -2 * ll + log_det_info(m, lam, d, K, Z, T) for m, (ll, lam) in fits.items()}
            else:
                score = {m: -2 * fits[m][0] + fp[m][c] for m in MODELS}
            hits[c] += min(score, key=score.get) == truth
    return {c: h / reps for c, h in hits.items()}


if __name__ == "__main__":
    import csv, os
    rng = np.random.default_rng(4)
    check_ml(rng)
    K, Z, T, REPS = 5, 5, 24, 1000
    var = {"ZIP": 0.5, "county": 0.3, "region": 0.3}
    truths = [("Depth 2: ZIP only", frozenset({"ZIP"})),
              ("Depth 3: ZIP + county", frozenset({"ZIP", "county"})),
              ("Depth 4: ZIP + county + region", frozenset(LEVELS)),
              ("Depth 4, middle missing: ZIP + region", frozenset({"ZIP", "region"}))]
    rows = []
    for label, truth in truths:
        for R in (4, 8, 15):
            rate = simulate(R, K, Z, T, var, truth, REPS, rng)
            for c, r in rate.items():
                rows.append({"truth": label, "regions": R, "criterion": c,
                             "pct_correct": round(100 * r, 1)})
            print(f"{label:40s} R={R:2d}", {c: round(100 * r, 1) for c, r in rate.items()})
    os.makedirs("results", exist_ok=True)
    with open("results/d1_depth_4level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
