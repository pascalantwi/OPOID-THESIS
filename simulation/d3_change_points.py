"""Pilot simulation for Direction 2 (nested random change points).

Each ZIP code j in county k has its own monthly series with one level shift:

    y_kjt = mu_kj + delta * 1{t >= tau_kj} + e_kjt,
    tau_kj = round(tau0 + b_k + c_kj),  b_k ~ N(0, s2_tauC),  c_kj ~ N(0, s2_tauZ),

with e_kjt Gaussian white noise or AR(1). This pilot does NOT fit the proposed
hierarchical model; it measures how well the simple alternatives work, which
is what the proposed model would have to improve on:

  * per-ZIP least-squares break dates (Bai-Perron with one break, each series
    estimated separately);
  * one common break date for all series (Bai, 2010);
  * a two-stage estimate of the timing variances: nested ANOVA (method of
    moments) applied to the per-ZIP estimated break dates.
"""
import numpy as np


def ls_break(Y, h):
    """Least-squares single break date for each row of Y (break at t >= tau).

    Returns the estimated tau per row and the SSR profile (rows x candidates).
    """
    n, T = Y.shape
    taus = np.arange(h, T - h + 1)
    cs, cs2 = np.cumsum(Y, axis=1), np.cumsum(Y ** 2, axis=1)
    tot, tot2 = cs[:, -1:], cs2[:, -1:]
    pre_n = taus
    pre_s, pre_s2 = cs[:, taus - 1], cs2[:, taus - 1]
    post_n = T - taus
    post_s, post_s2 = tot - pre_s, tot2 - pre_s2
    ssr = (pre_s2 - pre_s ** 2 / pre_n) + (post_s2 - post_s ** 2 / post_n)
    return taus[np.argmin(ssr, axis=1)], ssr, taus


def nested_mom(tau_hat, C, Z):
    """Method-of-moments timing variances from a balanced C x Z layout."""
    t = tau_hat.reshape(C, Z)
    mk = t.mean(axis=1)
    ms_within = ((t - mk[:, None]) ** 2).sum() / (C * (Z - 1))
    ms_between = Z * ((mk - t.mean()) ** 2).sum() / (C - 1)
    return max(0.0, (ms_between - ms_within) / Z), ms_within


def simulate(C, Z, T, tau0, sC, sZ, delta, phi, reps, rng, h=6):
    rows = {"rmse_zip": [], "rmse_common": [], "s2C_hat": [], "s2Z_hat": [], "exact": []}
    for _ in range(reps):
        tau = np.rint(tau0 + rng.normal(0, sC, (C, 1)) + rng.normal(0, sZ, (C, Z)))
        tau = np.clip(tau, h, T - h).astype(int).ravel()
        e = rng.normal(0, 1, (C * Z, T))
        if phi:                          # stationary AR(1), rescaled to unit variance
            e[:, 0] /= np.sqrt(1 - phi ** 2)
            for t in range(1, T):
                e[:, t] = phi * e[:, t - 1] + e[:, t]
            e *= np.sqrt(1 - phi ** 2)
        Y = rng.normal(0, 1, (C * Z, 1)) + delta * (np.arange(T)[None, :] >= tau[:, None]) + e
        tau_hat, ssr, taus = ls_break(Y, h)
        common = taus[np.argmin(ssr.sum(axis=0))]
        s2C_hat, s2Z_hat = nested_mom(tau_hat, C, Z)
        rows["rmse_zip"].append(np.sqrt(np.mean((tau_hat - tau) ** 2)))
        rows["rmse_common"].append(np.sqrt(np.mean((common - tau) ** 2)))
        rows["exact"].append(np.mean(tau_hat == tau))
        rows["s2C_hat"].append(s2C_hat)
        rows["s2Z_hat"].append(s2Z_hat)
    return {k: float(np.mean(v)) for k, v in rows.items()}


if __name__ == "__main__":
    import csv, os
    rng = np.random.default_rng(2014)
    C, Z, T, tau0, sC, sZ, REPS = 10, 10, 48, 24, 3.0, 2.0, 300
    out = []
    for phi in (0.0, 0.5):
        for delta in (0.5, 1.0, 2.0):
            r = simulate(C, Z, T, tau0, sC, sZ, delta, phi, REPS, rng)
            row = {"ar1_phi": phi, "jump_sd_units": delta,
                   "rmse_zip_months": round(r["rmse_zip"], 2),
                   "rmse_common_months": round(r["rmse_common"], 2),
                   "pct_dates_exact": round(100 * r["exact"], 1),
                   "true_s2_tau_county": sC ** 2, "est_s2_tau_county": round(r["s2C_hat"], 2),
                   "true_s2_tau_zip": sZ ** 2, "est_s2_tau_zip": round(r["s2Z_hat"], 2)}
            out.append(row); print(row)
    os.makedirs("results", exist_ok=True)
    with open("results/d3_change_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader(); w.writerows(out)
