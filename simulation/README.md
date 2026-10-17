# Pilot simulations

Small first simulations for the two thesis directions. They are **pilots**: balanced designs, intercept-only means and modest numbers of replications. They show what the full simulation chapters would study and what the first results look like. They are not final results.

Run with Python 3 (numpy, scipy, matplotlib):

```
python3 d1_bic_levels.py      # Direction 1, three levels -> results/d1_selection.csv
python3 d1_depth_4level.py    # Direction 1, depths 2-4   -> results/d1_depth_4level.csv
python3 d3_change_points.py   # Direction 2 -> results/d3_change_points.csv
python3 make_figures.py       # figures/*.png
```

## Direction 1: selecting levels with BIC

**Design.** Months (T = 24) within ZIP codes (Z = 10 per county) within C = 5, 10 or 30 counties; 1,000 data sets per condition; residual variance 1, ZIP variance 0.5. Four candidate models are fitted by maximum likelihood: no random effects (M0), ZIP only (MZ), county only (MC), and county plus ZIP (M3). The likelihood is computed exactly from the three ANOVA strata of the balanced design; the code checks the closed-form ML fit against a numerical optimiser.

**Criteria compared.** AIC; BIC with total N; BIC with the number of top-level clusters; the leading-order level-specific BIC (proposal equation 4); and a BIC whose penalty is the exact log-determinant of the Fisher information at each model's own fit (the Laplace approximation, proposal equation 2).

![Direction 1 results](figures/d1_selection.png)

**Percentage of data sets in which the true structure was selected**

| Scenario (truth) | Counties | AIC | BIC (N) | BIC (top clusters) | BIC (level-specific) | BIC (exact information) |
|---|---|---|---|---|---|---|
| A: county and ZIP real (M3) | 5 | 79.1 | 53.3 | 100.0 | 100.0 | 69.4 |
| | 10 | 96.5 | 84.7 | 100.0 | 99.6 | 90.8 |
| | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| B: no county effect (MZ) | 5 | 96.1 | 99.9 | 0.0 | 0.0 | 99.2 |
| | 10 | 96.2 | 99.9 | 0.0 | 61.6 | 99.1 |
| | 30 | 94.8 | 99.9 | 0.0 | 88.9 | 99.9 |
| C: weak county effect (M3, county variance 0.05) | 5 | 21.4 | 3.6 | 100.0 | 100.0 | 8.9 |
| | 10 | 41.0 | 8.7 | 100.0 | 82.6 | 17.7 |
| | 30 | 81.4 | 31.1 | 100.0 | 88.3 | 44.1 |

**What the pilot shows**

1. **BIC with total N drops real county levels when counties are few:** it finds the true structure in only 53% of data sets with 5 counties (scenario A).
2. **The leading-order level-specific penalty (equation 4) fails when a level is absent.** With fewer counties than ZIP codes per county, adding the county level moves the intercept's penalty from log(#ZIPs) down to log(#counties). The larger model then has a *smaller* penalty and is always chosen, so the criterion never detects a missing county level (0% in scenario B with 5 counties). BIC with top-level clusters fails for the same reason. Equation 4 was derived assuming every variance component is positive. **The derivation therefore has to treat the boundary case (a variance equal to zero) explicitly**, which is the central theoretical task of Direction 1.
3. **Using the exact information instead of leading-order log n terms repairs this.** That criterion keeps false county levels out (99% correct in scenario B) and finds real county levels more often than BIC with total N (69% vs 53% with 5 counties; 91% vs 85% with 10).
4. **No criterion detects a weak county effect with few counties.** This trade-off between power and false inclusion is what the full simulation chapter would map.

## Direction 1 with two, three and four levels

**Design.** Months (T = 24) within ZIP codes (5 per county) within counties (5 per region) within R = 4, 8 or 15 regions; variances: residual 1, ZIP 0.5, county 0.3, region 0.3; 1,000 data sets per condition. The true structure has depth 2 (ZIP only), 3 (ZIP + county), 4 (ZIP + county + region), or 4 with the middle level missing (ZIP + region). All 8 candidate structures are fitted by exact ML (checked against a numerical optimiser) and the same five criteria choose among them.

![Direction 1, depths 2-4](figures/d1_depth_4level.png)

**Percentage of data sets in which the true structure was selected**

| True structure | Regions | AIC | BIC (N) | BIC (top clusters) | BIC (level-specific) | BIC (exact information) |
|---|---|---|---|---|---|---|
| Depth 2: ZIP only | 4 | 91.9 | 99.8 | 0.0 | 0.0 | 98.9 |
| | 8 | 92.1 | 99.9 | 0.0 | 0.0 | 99.7 |
| | 15 | 90.6 | 99.9 | 0.0 | 0.0 | 99.9 |
| Depth 3: ZIP + county | 4 | 94.7 | 82.1 | 0.0 | 0.0 | 91.7 |
| | 8 | 95.3 | 98.6 | 0.0 | 85.5 | 98.6 |
| | 15 | 94.6 | 100.0 | 0.0 | 89.3 | 99.1 |
| Depth 4: ZIP + county + region | 4 | 47.2 | 9.0 | 97.8 | 94.5 | 33.7 |
| | 8 | 80.7 | 39.4 | 99.7 | 90.2 | 66.7 |
| | 15 | 96.8 | 74.5 | 100.0 | 97.8 | 90.5 |
| Depth 4, middle missing: ZIP + region | 4 | 83.0 | 74.2 | 91.3 | 97.4 | 80.9 |
| | 8 | 95.0 | 95.3 | 95.8 | 98.2 | 97.2 |
| | 15 | 91.9 | 99.9 | 94.5 | 98.3 | 99.7 |

**What it shows**

1. **When the true model is shallow (depth 2 or 3), the leading-order penalties fail completely:** BIC with top-level clusters and the level-specific BIC add levels that do not exist (0% correct with 4 regions). Their high scores for depth 4 come from this same bias toward extra levels, not from skill.
2. **BIC with total N is reliable for shallow truths but misses a real region level when regions are few:** 9% correct with 4 regions and 39% with 8.
3. **The exact-information BIC is the only BIC that does well in every scenario with enough regions** (91-100% with 15 regions). It recovers depth 4 far more often than BIC with total N (34% vs 9% with 4 regions; 67% vs 39% with 8) while staying near 99% for shallow truths.
4. **With only 4 regions, no criterion recovers depth 4 reliably.** This is the few-top-level-units problem the thesis targets.

## Direction 2: nested random change points

**Design.** C = 10 counties × Z = 10 ZIP codes, T = 48 months, one level shift per ZIP code. Break dates are nested: county shift SD 3 months and ZIP shift SD 2 months around month 24 (true timing variances 9 and 4). The jump size is 0.5, 1 or 2 noise SDs, with white-noise or AR(1) (phi = 0.5) errors; 300 data sets per condition.

The pilot does **not** fit the proposed hierarchical model. It measures the simple alternatives the model would have to beat: per-ZIP least-squares break dates, one common break date, and a two-stage estimate of the timing variances (nested ANOVA on the per-ZIP estimated dates).

![Direction 2 results](figures/d3_change_points.png)

| Errors | Jump (SD) | RMSE per-ZIP (months) | RMSE common break | Exact dates (%) | County timing var. (true 9) | ZIP timing var. (true 4) |
|---|---|---|---|---|---|---|
| white noise | 0.5 | 9.90 | 3.83 | 9.1 | 3.18 | 95.23 |
| | 1.0 | 5.46 | 3.53 | 26.0 | 6.51 | 33.29 |
| | 2.0 | 1.41 | 3.50 | 62.8 | 8.71 | 6.10 |
| AR(1), 0.5 | 0.5 | 11.35 | 4.22 | 6.1 | 2.27 | 121.24 |
| | 1.0 | 8.60 | 3.64 | 17.5 | 3.68 | 73.44 |
| | 2.0 | 3.65 | 3.52 | 54.3 | 7.84 | 17.24 |

**What the pilot shows**

1. **Per-ZIP estimates are reliable only for large jumps.** With a jump of 2 SDs the break-date RMSE is 1.4 months; with 0.5 SD it is almost 10 months.
2. **A common break date ignores real differences in timing.** Its error stays near 3.5 months whatever the jump size, because ZIP codes truly break at different times.
3. **The two-stage approach badly misstates the timing variances.** Estimation error in each ZIP's date is counted as real ZIP-level variation: the estimated ZIP timing variance is 95 instead of 4 for small jumps. Meanwhile the county-level variance is underestimated. Autocorrelation makes this worse.
4. **This is the gap the proposed model targets.** A hierarchical model that estimates break dates and timing variances jointly can separate estimation noise from true variation in timing and borrow strength across ZIP codes in the same county.

## Limitations of these pilots

Balanced designs only; intercept-only means; Gaussian errors; no covariates; modest replication counts; and the proposed change-point model itself is not yet fitted. The full simulation chapters would relax each of these.
