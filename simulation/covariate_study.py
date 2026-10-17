"""Level selection with covariates (Direction 1, four levels, ARCOS-shaped design).

Model for log MME per 1,000 in month t of ZIP j (county k, region g):

    y_jt = mu + b_inc * income_j + b_rur * rural_j + g_trend * trend_t + g_post * post_t
           + region_g + county_k + zip_j + e_jt

ZIP-level covariates: income (standardised MEAN_INCOME_$; clustered by county
and region) and rurality (from RUCA_NUMBER; the share of rural ZIP codes differs
by region). Month-level covariates (from TRANSACTION_DATE): linear trend and a
post-October-2014 indicator (hydrocodone rescheduling). Same design as the main
study: 7 regions, 67 counties, 1,483 ZIP codes, 60 months.

Exact ML with fixed effects profiled out: month-level covariates are the same
for every ZIP, so after centring they lie in the within-ZIP stratum and are
estimated by pooled OLS on within-ZIP deviations. ZIP-level covariates enter
both the within-county ZIP contrasts (variance a = s2_Z + s2_e/T) and the county
means (region blocks diag(b_k) + s2_R J), and are estimated by GLS across the
two pieces. The code is checked against a dense-matrix likelihood.
"""
import csv
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_design_study as rd  # noqa: E402

LEVELS = rd.LEVELS
MODELS = rd.MODELS
LOG2PI = np.log(2 * np.pi)
T = 60
REPS = int(os.environ.get("REPS", 500))


# ---------------------------------------------------------------- design + covariates
def make_covariates(design, rural_spread, seed=2014):
    rng = np.random.default_rng(seed)
    d = design
    inc = (0.6 * rng.normal(size=d.R)[d.rc][d.cz] + 0.6 * rng.normal(size=d.K)[d.cz]
           + rng.normal(size=d.J))
    inc = (inc - inc.mean()) / inc.std()
    p_rural = np.linspace(0.5 - rural_spread, 0.5 + rural_spread, d.R)
    rng.shuffle(p_rural)
    rural = (rng.random(d.J) < p_rural[d.rc][d.cz]).astype(float)
    t = np.arange(T)
    time = np.column_stack([(t - t.mean()) / T, (t >= T // 2) - 0.5])   # trend, post-2014
    time -= time.mean(axis=0)
    return np.column_stack([inc, rural]), time


# ---------------------------------------------------------------- sufficient statistics
def suff(design, Y, Xz, time):
    """Y: J x T outcomes; Xz: J x p ZIP covariates (p may be 0); time: T x q."""
    d = design
    m = Y.mean(axis=1)
    dev = Y - m[:, None]
    tt = time.T @ time
    gamma = np.linalg.solve(d.J * tt, time.T @ dev.sum(axis=0)) if time.shape[1] else np.zeros(0)
    W = ((dev - (time @ gamma)[None, :]) ** 2).sum()
    M = np.bincount(d.cz, weights=m, minlength=d.K) / d.n_k
    Xbar = np.column_stack([np.bincount(d.cz, weights=Xz[:, i], minlength=d.K) / d.n_k
                            for i in range(Xz.shape[1])]) if Xz.shape[1] else np.zeros((d.K, 0))
    yc = m - M[d.cz]
    Zc = Xz - Xbar[d.cz]
    return {"W": W, "yc": yc, "Zc": Zc, "M": M, "Xbar": Xbar, "time_info": d.J * tt}


def gls_pieces(design, st, theta):
    d = design
    a, b, S1, sr = rd.region_pieces(d, theta)
    denom = 1 + sr * S1
    Xm = np.column_stack([np.ones(d.K), st["Xbar"]])
    p1 = Xm.shape[1]
    Ub = np.zeros((d.R, p1))
    np.add.at(Ub, d.rc, Xm / b[:, None])
    SMb = np.bincount(d.rc, weights=st["M"] / b, minlength=d.R)
    A = (Xm / b[:, None]).T @ Xm - (sr / denom)[:, None, None].__mul__(
        Ub[:, :, None] * Ub[:, None, :]).sum(axis=0)
    rhs = (Xm / b[:, None]).T @ st["M"] - ((sr / denom) * SMb) @ Ub
    A[1:, 1:] += st["Zc"].T @ st["Zc"] / a
    rhs[1:] += st["Zc"].T @ st["yc"] / a
    return a, b, S1, sr, denom, Xm, A, rhs


def loglik(design, st, theta):
    d = design
    se = theta[0]
    a, b, S1, sr = rd.region_pieces(d, theta)
    if se <= 0 or a <= 0 or np.any(b <= 0) or np.any(1 + sr * S1 <= 0):
        return -np.inf
    a, b, S1, sr, denom, Xm, A, rhs = gls_pieces(d, st, theta)
    coef = np.linalg.solve(A, rhs)
    rc = st["yc"] - st["Zc"] @ coef[1:]
    r = st["M"] - Xm @ coef
    Sr = np.bincount(d.rc, weights=r / b, minlength=d.R)
    quad = (r ** 2 / b).sum() - (sr * Sr ** 2 / denom).sum()
    nw, nc = d.J * (T - 1), d.J - d.K
    return -0.5 * (d.N * LOG2PI + nw * np.log(se) + st["W"] / se
                   + nc * np.log(a) + (rc ** 2).sum() / a
                   + d.J * np.log(T) + np.log(d.n_k).sum()
                   + np.log(b).sum() + np.log(denom).sum() + quad)


def fit(design, st, present):
    free = [0] + [1 + LEVELS.index(l) for l in LEVELS if l in present]
    se0 = st["W"] / (design.J * (T - 1))

    def full(p):
        th = np.zeros(4); th[free] = p
        return th

    def nll(p):
        ll = loglik(design, st, full(p))
        return 1e300 if not np.isfinite(ll) else -ll

    best = None
    for start in (0.05, 0.5):
        r = minimize(nll, [se0] + [start] * (len(free) - 1), method="L-BFGS-B",
                     bounds=[(1e-8, None)] + [(0, None)] * (len(free) - 1))
        if best is None or r.fun < best.fun:
            best = r
    th = full(best.x)
    return loglik(design, st, th), th


def log_det_info(design, st, theta, present):
    """log det expected information: fixed effects (GLS + time) and variance components."""
    d = design
    *_, A, _ = gls_pieces(d, st, theta)
    ld_fixed = np.linalg.slogdet(A)[1]
    if st["time_info"].size:
        ld_fixed += np.linalg.slogdet(st["time_info"] / theta[0])[1]
    ld_mu_and_var = rd.log_det_expected_info(d, theta, present)
    # rd's version includes log(info for mu) from the intercept-only GLS; replace it by the GLS block above
    a, b, S1, sr = rd.region_pieces(d, theta)
    info_mu = (S1 / (1 + sr * S1)).sum()
    return ld_fixed + ld_mu_and_var - np.log(info_mu)


# ---------------------------------------------------------------- selection
def choose(design, st):
    fits = {m: fit(design, st, m) for m in MODELS}
    nfix = 1 + st["Zc"].shape[1] + st["time_info"].shape[0]
    units = {"e": design.N, "ZIP": design.J, "county": design.K, "region": design.R}
    m2ll = {m: -2 * fits[m][0] for m in MODELS}
    k = {m: nfix + 1 + len(m) for m in MODELS}
    top = {m: max((l for l in LEVELS if l in m), key=LEVELS.index, default="e") for m in MODELS}
    out = {
        "AIC": min(MODELS, key=lambda m: m2ll[m] + 2 * k[m]),
        "Standard BIC (N)": min(MODELS, key=lambda m: m2ll[m] + k[m] * np.log(units["e"])),
        "New BIC (exact information)": min(
            MODELS, key=lambda m: m2ll[m] + log_det_info(design, st, fits[m][1], m)),
    }
    cur = frozenset(LEVELS)
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
    return out


SCENARIOS = [
    # label, variances, rural effect, spread of rural share across regions, fit ZIP covariates?
    ("No region effect",                         {"ZIP": .5, "county": .3, "region": 0.0},  0.5, 0.2, True),
    ("Small region effect",                      {"ZIP": .5, "county": .3, "region": 0.05}, 0.5, 0.2, True),
    ("Medium region effect",                     {"ZIP": .5, "county": .3, "region": 0.3},  0.5, 0.2, True),
    ("Region gap due to rurality: covariates IN", {"ZIP": .5, "county": .3, "region": 0.0}, 1.5, 0.45, True),
    ("Region gap due to rurality: covariates OUT", {"ZIP": .5, "county": .3, "region": 0.0}, 1.5, 0.45, False),
]
BETA_INC, G_TREND, G_POST = -0.2, -0.5, -0.3

_design = None


def _init():
    global _design
    rc, cz = rd.make_design()
    _design = rd.Design(rc, cz, T)


def one_rep(args):
    s_idx, seed = args
    label, var, b_rur, spread, use_cov = SCENARIOS[s_idx]
    d = _design
    Xz, time = make_covariates(d, spread)
    rng = np.random.default_rng(seed)
    eff = (rng.normal(0, np.sqrt(var["region"]), d.R)[d.rc][d.cz]
           + rng.normal(0, np.sqrt(var["county"]), d.K)[d.cz]
           + rng.normal(0, np.sqrt(var["ZIP"]), d.J)
           + Xz @ np.array([BETA_INC, b_rur]))
    Y = eff[:, None] + (time @ np.array([G_TREND, G_POST]))[None, :] + rng.normal(0, 1, (d.J, T))
    st = suff(d, Y, Xz if use_cov else np.zeros((d.J, 0)), time)
    return s_idx, choose(d, st)


def self_test():
    """Compare the structured likelihood with a dense-matrix Gaussian likelihood."""
    rng = np.random.default_rng(3)
    rc = np.array([0, 0, 1, 1, 1]); cz = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4])
    global T
    T_old, T = T, 4
    d = rd.Design(rc, cz, T)
    Xz = np.column_stack([rng.normal(size=d.J), rng.integers(0, 2, d.J)]).astype(float)
    t = np.arange(T); time = np.column_stack([(t - t.mean()) / T, (t >= 2) - 0.5]); time -= time.mean(0)
    Y = rng.normal(size=(d.J, T)) + Xz[:, :1] * 0.3
    st = suff(d, Y, Xz, time)
    theta = np.array([0.8, 0.4, 0.3, 0.2])
    # dense: y = X b + noise, V from the variances; ML profile over b
    X = np.column_stack([np.ones(d.N), np.repeat(Xz, T, axis=0), np.tile(time, (d.J, 1))])
    Zr = np.repeat(np.eye(d.R)[rc][cz], T, axis=0); Zc = np.repeat(np.eye(d.K)[cz], T, axis=0)
    Zz = np.repeat(np.eye(d.J), T, axis=0)
    V = (theta[0] * np.eye(d.N) + theta[1] * Zz @ Zz.T + theta[2] * Zc @ Zc.T + theta[3] * Zr @ Zr.T)
    Vi = np.linalg.inv(V); y = Y.ravel()
    bhat = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y); r = y - X @ bhat
    dense = -0.5 * (d.N * LOG2PI + np.linalg.slogdet(V)[1] + r @ Vi @ r)
    assert abs(dense - loglik(d, st, theta)) < 1e-8, (dense, loglik(d, st, theta))
    # information for (mu, ZIP covariates, time) equals X' V^-1 X
    *_, A, _ = gls_pieces(d, st, theta)
    full_fixed = np.linalg.slogdet(X.T @ Vi @ X)[1]
    assert abs(full_fixed - (np.linalg.slogdet(A)[1] + np.linalg.slogdet(st["time_info"] / theta[0])[1])) < 1e-8
    T = T_old


if __name__ == "__main__":
    self_test()
    print("self-test passed: structured likelihood = dense-matrix likelihood")
    jobs = [(s, 7919 * (s + 1) + r) for s in range(len(SCENARIOS)) for r in range(REPS)]
    with Pool(initializer=_init) as pool:
        out = pool.map(one_rep, jobs, chunksize=8)
    rows = []
    for s, (label, var, *_rest) in enumerate(SCENARIOS):
        truth = frozenset(l for l in LEVELS if var[l] > 0)
        res = [c for i, c in out if i == s]
        for meth in res[0]:
            picks = [c[meth] for c in res]
            rows.append({"scenario": label, "true_structure": rd.name(truth), "method": meth,
                         "pct_correct": round(100 * np.mean([p == truth for p in picks]), 1),
                         "pct_says_region_matters": round(100 * np.mean(["region" in p for p in picks]), 1)})
        print(f"{label:44s}", {r["method"]: (r["pct_correct"], r["pct_says_region_matters"])
                                for r in rows if r["scenario"] == label}, flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/covariate_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
