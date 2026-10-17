"""Pilot simulation for Direction 1: selecting the levels of a nested model with BIC.

Balanced three-level design: T months within Z ZIP codes within C counties,
intercept-only mean, Gaussian random intercepts:

    y_kjt = mu + u_k + w_kj + e_kjt,
    u_k ~ N(0, s2_C),  w_kj ~ N(0, s2_Z),  e_kjt ~ N(0, s2_e).

Four candidate models are fitted by maximum likelihood:
    M0  no random effects        MZ  ZIP level only
    MC  county level only        M3  county and ZIP levels
and selected with AIC, BIC with total N, BIC with the number of top-level
clusters, the leading-order level-specific BIC of the proposal (equation 4),
and a BIC whose penalty is the exact log det of the Fisher information at each
model's own ML fit (the Laplace approximation, proposal equation 2).

The likelihood is computed exactly from the three ANOVA strata of the balanced
design (eigenvalues lambda_1 <= lambda_2 <= lambda_3), and the order-restricted
ML estimate is the weighted isotonic regression of the stratum mean squares.
"""
import itertools
import numpy as np
from scipy.optimize import minimize

LOG2PI = np.log(2 * np.pi)


def strata(y):
    """Sums of squares and degrees of freedom of the three strata."""
    C, Z, T = y.shape
    m_kj = y.mean(axis=2)
    m_k = y.mean(axis=(1, 2))
    m = y.mean()
    W = ((y - m_kj[..., None]) ** 2).sum()            # months within ZIP
    B = T * ((m_kj - m_k[:, None]) ** 2).sum()         # ZIPs within county
    K = Z * T * ((m_k - m) ** 2).sum()                 # counties (mean profiled out)
    return np.array([W, B, K]), np.array([C * Z * (T - 1), C * (Z - 1), C])


def pava(s, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
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


def fit(S, d, groups):
    """ML fit when strata in the same group share one eigenvalue.

    groups: list of lists of stratum indices, ordered by increasing eigenvalue.
    Returns the maximised log-likelihood and the fitted eigenvalue of each stratum.
    """
    SS = np.array([S[g].sum() for g in groups])
    dd = np.array([d[g].sum() for g in groups])
    lam = pava(SS / dd, dd)
    N = d.sum()
    ll = -0.5 * (N * LOG2PI + (dd * np.log(lam)).sum() + (SS / lam).sum())
    lam_strata = np.empty(3)
    for g, l in zip(groups, lam):
        lam_strata[g] = l
    return ll, lam_strata


def max_loglik(S, d, groups):
    return fit(S, d, groups)[0]


# d lambda_m / d theta for each model: rows = strata (within ZIP, ZIP, county),
# columns = variance parameters present in the model
def jacobian(model, Z, T):
    cols = {"se": [1, 1, 1], "sz": [0, T, T], "sc": [0, 0, Z * T]}
    params = {"M0": ["se"], "MZ": ["se", "sz"], "MC": ["se", "sc"], "M3": ["se", "sz", "sc"]}[model]
    return np.array([cols[p] for p in params], dtype=float).T


def log_det_info(model, lam, d, Z, T):
    """log det of the Fisher information at the fitted values (exact, balanced design).

    Mean and variance parameters are orthogonal; the intercept's information is
    N / lambda_3 because the vector of ones lies in the county stratum.
    """
    A = jacobian(model, Z, T)
    I_theta = A.T @ np.diag(d / (2 * lam ** 2)) @ A
    return np.log(d.sum() / lam[2]) + np.linalg.slogdet(I_theta)[1]


MODELS = {            # strata grouping, number of parameters
    "M0": ([[0, 1, 2]], 2),
    "MZ": ([[0], [1, 2]], 3),
    "MC": ([[0, 1], [2]], 3),
    "M3": ([[0], [1], [2]], 4),
}


def penalties(C, Z, T):
    N, nZ = C * Z * T, C * Z
    lN, lZ, lC = np.log(N), np.log(nZ), np.log(C)
    return {
        "AIC":        {m: 2 * k for m, (_, k) in MODELS.items()},
        "BIC (N)":    {m: k * lN for m, (_, k) in MODELS.items()},
        "BIC (top clusters)": {"M0": 2 * lN, "MZ": 3 * lZ, "MC": 3 * lC, "M3": 4 * lC},
        # level-specific penalty (proposal eq. 4): each parameter pays the log of
        # the number of units at the level that carries its information
        "BIC (level-specific)": {"M0": 2 * lN, "MZ": lN + 2 * lZ,
                                 "MC": lN + 2 * lC, "M3": lN + lZ + 2 * lC},
    }


def check_ml_against_optimizer(rng):
    """Confirm the closed-form ML (PAVA) equals a numerical optimum."""
    C, Z, T = 6, 5, 8
    for _ in range(20):
        y = (rng.normal(0, 0.5, (C, 1, 1)) + rng.normal(0, 0.7, (C, Z, 1))
             + rng.normal(0, 1, (C, Z, T)))
        S, d = strata(y)

        def nll(p):
            se, sz, sc = p
            lam = np.array([se, se + T * sz, se + T * sz + Z * T * sc])
            return 0.5 * (d.sum() * LOG2PI + (d * np.log(lam)).sum() + (S / lam).sum())

        best = min((minimize(nll, x0, bounds=[(1e-8, None), (0, None), (0, None)],
                             method="L-BFGS-B") for x0 in
                    itertools.product([0.5, 2.0], [0.0, 0.5], [0.0, 0.5])),
                   key=lambda r: r.fun)
        assert abs(-best.fun - max_loglik(S, d, MODELS["M3"][0])) < 1e-5


def simulate(C, Z, T, s2C, s2Z, reps, rng):
    truth = {(True, True): "M3", (False, True): "MZ",
             (True, False): "MC", (False, False): "M0"}[(s2C > 0, s2Z > 0)]
    pen = penalties(C, Z, T)
    hits = {c: 0 for c in pen}
    for _ in range(reps):
        y = (rng.normal(0, np.sqrt(s2C), (C, 1, 1))
             + rng.normal(0, np.sqrt(s2Z), (C, Z, 1))
             + rng.normal(0, 1.0, (C, Z, T)))
        S, d = strata(y)
        fits = {m: fit(S, d, g) for m, (g, _) in MODELS.items()}
        m2ll = {m: -2 * ll for m, (ll, _) in fits.items()}
        p_info = {m: log_det_info(m, lam, d, Z, T) for m, (_, lam) in fits.items()}
        for crit, p in list(pen.items()) + [("BIC (exact information)", p_info)]:
            chosen = min(MODELS, key=lambda m: m2ll[m] + p[m])
            hits[crit] = hits.get(crit, 0) + (chosen == truth)
    return truth, {c: h / reps for c, h in hits.items()}


if __name__ == "__main__":
    import csv, os
    rng = np.random.default_rng(2026)
    check_ml_against_optimizer(rng)
    Z, T, REPS = 10, 24, 1000
    scenarios = [("A: county and ZIP real", 0.3, 0.5),
                 ("B: no county effect", 0.0, 0.5),
                 ("C: weak county effect", 0.05, 0.5)]
    rows = []
    for label, s2C, s2Z in scenarios:
        for C in (5, 10, 30):
            truth, rate = simulate(C, Z, T, s2C, s2Z, REPS, rng)
            for crit, r in rate.items():
                rows.append({"scenario": label, "counties": C, "true_model": truth,
                             "criterion": crit, "pct_correct": round(100 * r, 1)})
            print(label, C, {k: round(100 * v, 1) for k, v in rate.items()})
    os.makedirs("results", exist_ok=True)
    with open("results/d1_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
