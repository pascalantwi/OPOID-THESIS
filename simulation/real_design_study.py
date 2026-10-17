"""Main pilot study for Direction 1: which method best identifies the levels that matter?

Design shaped like the ARCOS data: 7 regions, 67 counties (unequal numbers per
region) and 1,483 ZIP codes (unequal numbers per county), each observed for T
months. Model (intercept only):

    y = mu + region + county + ZIP + error,   variances s2_R, s2_C, s2_Z, s2_e.

Each simulated data set is fitted under all 8 level structures (every subset of
{ZIP, county, region}) by exact maximum likelihood, and six methods choose a
structure:

    AIC; standard BIC (penalty k log N, as in R/lme4); BIC with the number of
    top-level clusters (SAS-style); the leading-order level-specific BIC
    (proposal eq. 4); backward likelihood-ratio testing with the boundary
    (50:50 chi-square mixture) correction at 5%; and the new exact-information
    BIC, whose penalty is log det of the expected Fisher information at each
    model's own fit (the Laplace approximation, proposal eq. 2).

Exact likelihood for unequal group sizes: the data split into independent
Gaussian pieces -- months within ZIP (variance s2_e), ZIP contrasts within
county (variance a = s2_Z + s2_e/T), and county means, which within a region
have covariance diag(a/n_k + s2_C) + s2_R J. Determinants and quadratic forms
of the last piece use the matrix determinant lemma and Sherman-Morrison.
"""
import csv
import itertools
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

LEVELS = ("ZIP", "county", "region")
MODELS = [frozenset(s) for r in range(4) for s in itertools.combinations(LEVELS, r)]
PARAMS = ("e", "ZIP", "county", "region")
LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------- design
def make_design(seed=1483, R=7, n_counties=67, n_zips=1483):
    """Unequal design with the real totals (synthetic group sizes)."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0, 0.4, R)
    cpr = np.maximum(3, np.round(w / w.sum() * n_counties)).astype(int)
    cpr[np.argmax(cpr)] += n_counties - cpr.sum()
    w = rng.lognormal(0, 0.8, n_counties)
    zpc = np.maximum(2, np.round(w / w.sum() * n_zips)).astype(int)
    zpc[np.argmax(zpc)] += n_zips - zpc.sum()
    region_of_county = np.repeat(np.arange(R), cpr)
    county_of_zip = np.repeat(np.arange(n_counties), zpc)
    return region_of_county, county_of_zip


class Design:
    def __init__(self, region_of_county, county_of_zip, T):
        self.rc, self.cz, self.T = region_of_county, county_of_zip, T
        self.R, self.K, self.J = region_of_county.max() + 1, len(region_of_county), len(county_of_zip)
        self.n_k = np.bincount(county_of_zip, minlength=self.K)       # ZIPs per county
        self.N = self.J * T


# ---------------------------------------------------------------- data
def simulate(design, var, rng):
    """Return sufficient statistics: within-ZIP SS, ZIP means."""
    d, T = design, design.T
    reg = rng.normal(0, np.sqrt(var["region"]), d.R)
    cty = rng.normal(0, np.sqrt(var["county"]), d.K)
    zip_eff = rng.normal(0, np.sqrt(var["ZIP"]), d.J)
    e = rng.normal(0, np.sqrt(var["e"]), (d.J, T))
    y = (reg[d.rc][d.cz] + cty[d.cz] + zip_eff)[:, None] + e
    m = y.mean(axis=1)
    W = ((y - m[:, None]) ** 2).sum()
    return W, m


def suff(design, W, m):
    d = design
    M = np.bincount(d.cz, weights=m, minlength=d.K) / d.n_k          # county means
    Q = ((m - M[d.cz]) ** 2).sum()                                    # ZIP contrasts SS
    return W, Q, M


# ---------------------------------------------------------------- likelihood
def region_pieces(design, theta):
    se, sz, sc, sr = theta
    a = sz + se / design.T
    b = a / design.n_k + sc                                           # county-mean variances
    S1 = np.bincount(design.rc, weights=1 / b, minlength=design.R)    # sum 1/b per region
    return a, b, S1, sr


def loglik(design, stats, theta, mu=None):
    """Log-likelihood; mu=None profiles the mean out (GLS). Returns (ll, mu)."""
    W, Q, M = stats
    d = design
    se = theta[0]
    a, b, S1, sr = region_pieces(d, theta)
    if se <= 0 or a <= 0 or np.any(b <= 0) or np.any(1 + sr * S1 <= 0):
        return -np.inf, mu
    SM = np.bincount(d.rc, weights=M / b, minlength=d.R)              # sum M/b per region
    denom = 1 + sr * S1
    if mu is None:
        mu = (SM / denom).sum() / (S1 / denom).sum()
    x = M - mu
    Sx = np.bincount(d.rc, weights=x / b, minlength=d.R)
    quad = (x ** 2 / b).sum() - (sr * Sx ** 2 / denom).sum()
    logdet = np.log(b).sum() + np.log(denom).sum()
    nw, nc = d.J * (d.T - 1), d.J - d.K
    ll = -0.5 * (d.N * LOG2PI + nw * np.log(se) + W / se            # months within ZIP
                 + nc * np.log(a) + Q / a                             # ZIP contrasts
                 + d.J * np.log(d.T) + np.log(d.n_k).sum()            # Jacobians
                 + logdet + quad)                                     # county means
    return ll, mu


def fit(design, stats, present):
    """ML fit of one structure; variances of absent levels fixed at zero."""
    free = [0] + [1 + LEVELS.index(l) for l in LEVELS if l in present]
    W, Q, M = stats
    se0 = W / (design.J * (design.T - 1))

    def full(p):
        th = np.zeros(4); th[free] = p
        return th

    def nll(p):
        ll, _ = loglik(design, stats, full(p))
        return 1e300 if not np.isfinite(ll) else -ll

    best = None
    for start in (0.05, 0.5):
        x0 = [se0] + [start] * (len(free) - 1)
        r = minimize(nll, x0, method="L-BFGS-B",
                     bounds=[(1e-8, None)] + [(0, None)] * (len(free) - 1))
        if best is None or r.fun < best.fun:
            best = r
    theta = full(best.x)
    ll, mu = loglik(design, stats, theta)
    return ll, theta


def log_det_expected_info(design, theta, present):
    """log det of the expected Fisher information for (mu, s2_e, present variances)."""
    d, T = design, design.T
    se, sz, sc, sr = theta
    a, b, S1, _ = region_pieces(d, theta)
    idx = [0] + [1 + LEVELS.index(l) for l in LEVELS if l in present]
    I = np.zeros((4, 4))
    I[0, 0] += d.J * (T - 1) / (2 * se ** 2)                          # months within ZIP
    ga = np.array([1 / T, 1, 0, 0])                                   # d a / d theta
    I += (d.J - d.K) / (2 * a ** 2) * np.outer(ga, ga)                # ZIP contrasts
    info_mu = 0.0
    for g in range(d.R):                                              # county means, per region
        k = np.where(d.rc == g)[0]
        n = d.n_k[k]
        Sig = np.diag(a / n + sc) + sr * np.ones((len(k), len(k)))
        Si = np.linalg.inv(Sig)
        dS = [np.diag(1 / (T * n)), np.diag(1 / n), np.eye(len(k)), np.ones((len(k), len(k)))]
        P = [Si @ D for D in dS]
        I += 0.5 * np.array([[np.sum(P[i] * P[j].T) for j in range(4)] for i in range(4)])
        info_mu += Si.sum()
    sub = I[np.ix_(idx, idx)]
    return np.log(info_mu) + np.linalg.slogdet(sub)[1]


# ---------------------------------------------------------------- selection
def name(present):
    return " + ".join(l for l in LEVELS if l in present) or "none"


def select_all(design, stats):
    fits = {m: fit(design, stats, m) for m in MODELS}
    units = {"e": design.N, "ZIP": design.J, "county": design.K, "region": design.R}
    choice = {}
    for crit in ("AIC", "Standard BIC (N)", "BIC (top clusters)", "BIC (level-specific, eq. 4)"):
        score = {}
        for m, (ll, _) in fits.items():
            k = 2 + len(m)
            top = max((l for l in LEVELS if l in m), key=LEVELS.index, default="e")
            pen = {"AIC": 2 * k,
                   "Standard BIC (N)": k * np.log(units["e"]),
                   "BIC (top clusters)": k * np.log(units[top]),
                   "BIC (level-specific, eq. 4)": np.log(units["e"]) + np.log(units[top])
                   + sum(np.log(units[l]) for l in m)}[crit]
            score[m] = -2 * ll + pen
        choice[crit] = min(score, key=score.get)
    score = {m: -2 * ll + log_det_expected_info(design, th, m) for m, (ll, th) in fits.items()}
    choice["New BIC (exact information)"] = min(score, key=score.get)
    # backward elimination with boundary-corrected likelihood ratio tests
    cur = frozenset(LEVELS)
    while cur:
        pvals = {}
        for l in cur:
            lr = max(0.0, 2 * (fits[cur][0] - fits[cur - {l}][0]))
            pvals[l] = 0.5 * chi2.sf(lr, 1) if lr > 0 else 1.0
        worst = max(pvals, key=pvals.get)
        if pvals[worst] > 0.05:
            cur = cur - {worst}
        else:
            break
    choice["Likelihood ratio tests"] = cur
    return choice


# ---------------------------------------------------------------- study
SCENARIOS = [
    ("1. Only ZIP matters",                 {"ZIP": 0.5, "county": 0.0, "region": 0.0}),
    ("2. ZIP + county",                     {"ZIP": 0.5, "county": 0.3, "region": 0.0}),
    ("3. All levels, small region effect",  {"ZIP": 0.5, "county": 0.3, "region": 0.05}),
    ("4. All levels, medium region effect", {"ZIP": 0.5, "county": 0.3, "region": 0.3}),
    ("5. All levels, large region effect",  {"ZIP": 0.5, "county": 0.3, "region": 1.0}),
    ("6. ZIP + region, no county",          {"ZIP": 0.5, "county": 0.0, "region": 0.3}),
]
METHODS = ["AIC", "Standard BIC (N)", "Likelihood ratio tests", "New BIC (exact information)",
           "BIC (top clusters)", "BIC (level-specific, eq. 4)"]
T, REPS = 60, 500


def one_rep(args):
    scen_idx, seed = args
    rc, cz = make_design()
    design = Design(rc, cz, T)
    var = dict(SCENARIOS[scen_idx][1], e=1.0)
    rng = np.random.default_rng(seed)
    W, m = simulate(design, var, rng)
    return scen_idx, select_all(design, suff(design, W, m))


def self_test():
    """Unequal-design likelihood and information agree with the balanced closed forms."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import d1_depth_4level as bal
    R, K, Z, T_ = 3, 4, 5, 6
    rc = np.repeat(np.arange(R), K); cz = np.repeat(np.arange(R * K), Z)
    design = Design(rc, cz, T_)
    rng = np.random.default_rng(0)
    y = (rng.normal(0, .6, (R, 1, 1, 1)) + rng.normal(0, .5, (R, K, 1, 1))
         + rng.normal(0, .7, (R, K, Z, 1)) + rng.normal(0, 1, (R, K, Z, T_)))
    yy = y.reshape(R * K * Z, T_)
    m = yy.mean(1); W = ((yy - m[:, None]) ** 2).sum()
    stats = suff(design, W, m)
    S, d = bal.strata(y)
    for present in MODELS:
        ll_b, lam = bal.fit(S, d, present)
        ll_u, th = fit(design, stats, present)
        assert abs(ll_b - ll_u) < 1e-4, (present, ll_b, ll_u)
        assert abs(bal.log_det_info(present, lam, d, K, Z, T_)
                   - log_det_expected_info(design, th, present)) < 1e-3, present


if __name__ == "__main__":
    self_test()
    rc, cz = make_design()
    print("design: regions", rc.max() + 1, "counties", len(rc), "ZIPs", len(cz),
          "| counties per region", np.bincount(rc).tolist(),
          "| ZIPs per county min/median/max", np.bincount(cz).min(),
          int(np.median(np.bincount(cz))), np.bincount(cz).max())
    jobs = [(s, 10_000 * s + r) for s in range(len(SCENARIOS)) for r in range(REPS)]
    with Pool() as pool:
        out = pool.map(one_rep, jobs, chunksize=10)
    rows = []
    for s, (label, var) in enumerate(SCENARIOS):
        truth = frozenset(l for l in LEVELS if var[l] > 0)
        res = [c for i, c in out if i == s]
        for meth in METHODS:
            picks = [c[meth] for c in res]
            correct = np.mean([p == truth for p in picks])
            added = np.mean([bool(p - truth) for p in picks])      # included a level that is not real
            missed = np.mean([bool(truth - p) for p in picks])     # left out a real level
            rows.append({"scenario": label, "true_structure": name(truth), "method": meth,
                         "pct_correct": round(100 * correct, 1),
                         "pct_added_false_level": round(100 * added, 1),
                         "pct_missed_real_level": round(100 * missed, 1)})
        print(label, {r["method"]: r["pct_correct"] for r in rows if r["scenario"] == label})
    os.makedirs("results", exist_ok=True)
    with open("results/main_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
