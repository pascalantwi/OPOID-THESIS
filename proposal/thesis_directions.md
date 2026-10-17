---
title: "Candidate Methodological Directions for the Master's Thesis"
subtitle: "Application: opioid distribution in ARCOS ZIP–month data"
author: "Fafa Tchorly — prepared for Dr. Michelle DeDeo"
date: "19 October 2026"
---

Following your guidance, I have narrowed the thesis to two methodological directions. Each starts from a classroom topic, asks a deeper statistical question that a simulation study can answer where the truth is known, and uses the opioid ZIP–month data as the application. Page 3 compares them and proposes combining them into one connected thesis.

# Direction 1: Information Criteria for Multilevel Models with Three or More Nested Levels

**Classroom starting point.** Linear mixed models estimated by ML/REML, and model selection with AIC and BIC.

**Deeper statistical question.** The usual criterion, BIC = −2 log L(θ̂) + k log n, assumes that the information about every parameter grows with the same sample size *n*. In nested data this fails: information about a parameter grows with the number of units at its own level (months, ZIP codes, counties, regions). Software packages also disagree on which *n* to use (total observations versus number of clusters). *What is the correct BIC penalty for three- and four-level nested models, and does it select the right levels when the top level has few units?*

**What is known.** Jones (2011) replaced *n* with an effective sample size for correlated data. Delattre et al. (2014) used the number of clusters for random effects and the total sample size for fixed effects. Lorah and Womack (2019) compared choices of *n* by simulation for two-level models and conjectured, without derivation, a level-specific penalty for three levels. Cho, Wu and Naveiras (2024) derived the penalty for level-specific selection in a *two-level* nested model and state that higher-level nested designs are left for future study. Greven and Kneib (2010) showed that marginal AIC is biased toward dropping random effects. I have not found a derivation for three or more levels.

**Theory to learn and develop.** The Laplace approximation to the marginal likelihood, log p(y) = log L(θ̂) − ½ log|I(θ̂)| + O(1); the spectral decomposition of nested covariance matrices; Fisher information for fixed effects and variance components; asymptotics as the numbers of units at each level grow; and behaviour at the boundary when a variance component is zero.

**Preliminary result (balanced three-level case; still to be checked against Cho et al.).** For months within ZIP codes within counties, the covariance matrix has three distinct eigenvalues, and the leading-order penalty becomes

> BIC~3~ = −2 log L + (p~1~ + 1) log N + (p~2~ + 1) log n~ZIP~ + (p~3~ + 1) log n~county~,

where p~1~, p~2~ and p~3~ are the numbers of month-, ZIP- and county-level fixed effects (intercept counted at county level). I confirmed this numerically against the exact Fisher information. One implication: when a level is removed, the penalties on the *other* parameters also change, which existing criteria ignore.

**Competing methods.** BIC with total *N*; BIC with number of clusters; Jones's effective sample size; AIC; conditional AIC; boundary-corrected and exact restricted likelihood ratio tests.

**Simulation study.** Generate data from known three- and four-level models, varying which variance components are zero, the number of top-level units (e.g., 4, 8, 15), units per level, intraclass correlations and balance. Measure the percentage of correctly selected structures, false-drop and false-keep rates per level, and coverage of fixed-effect confidence intervals after selection.

**Opioid application.** Log MME per 1,000 residents at the ZIP–month level: which of the ZIP, county and region levels the data support, the share of variation at each level, and whether conclusions about income and rurality depend on the levels kept. Sensitivity checks: AR(1) correlation across months, outcome scale, and ZIP codes that cross county lines.

**Key articles.** Jones (2011); Delattre, Lavielle and Poursat (2014); Greven and Kneib (2010); Lorah and Womack (2019); Cho, Wu and Naveiras (2024).

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Direction 2: Multilevel Random Change-Point Models for Nested Panel Data

**Classroom starting point.** Regression with a structural break: the Chow test at a known break date and piecewise linear regression.

**Deeper statistical question.** In practice the break date is unknown and need not be common to all units: ZIP codes may respond to a policy change at different times, and the timing may cluster within counties and regions. *Can break dates be modelled as random effects at several nested levels (ZIP within county within region), estimated jointly with the variance of timing at each level, and under what conditions are they identifiable?*

**What is known.** Andrews (1993) developed tests for a single break at an unknown date; Bai and Perron (1998) extended estimation and testing to multiple breaks; Bai (2010) estimated a common break date in panel data. Okui and Wang (2021) allowed breaks to differ across latent groups using grouped fixed effects and adaptive group fused lasso, and Lumsdaine, Okui and Wang (2023) allowed group membership itself to change at a break. Brilleman et al. (2017) fitted a Bayesian piecewise linear mixed model with a random change point at the individual level. I have not found a model with change points as random effects at several nested levels (this needs further confirmation).

**Theory to learn and develop.** The non-standard likelihood of piecewise regression with an unknown break (a discrete break date and a non-smooth likelihood); asymptotic properties of break-date estimators; hierarchical Bayesian modelling with marginalisation over discrete break dates; MCMC estimation; identifiability with short monthly series; and the effects of autocorrelation and heteroscedasticity.

**Competing methods.** Separate Bai–Perron estimation for each ZIP code; a common-break panel estimator (Bai, 2010); grouped heterogeneous breaks (Okui and Wang, 2021); a single-level random change-point model (Brilleman et al., 2017).

**Simulation study.** Generate break dates hierarchically, varying the numbers of ZIP codes, counties and regions, series length, break size, the spread of break dates at each level, and the presence of autocorrelation and heteroscedasticity. Measure bias and RMSE of break dates, coverage of credible intervals, recovery of the level-specific timing variances, and performance relative to the competing methods.

**Opioid application.** The DEA moved hydrocodone combination products from Schedule III to Schedule II, effective 6 October 2014 (79 FR 49661). The model would estimate when each ZIP code's hydrocodone supply shifted, how much the timing varies across ZIP codes, counties and regions, and whether rurality or income explains the timing. This requires data covering 2014, which I still need to confirm.

**Key articles.** Bai and Perron (1998); Bai (2010); Okui and Wang (2021); Lumsdaine, Okui and Wang (2023); Brilleman, Howe, Wolfe and Tilling (2017).

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Comparison and Preferred Plan

| | Direction 1: Multilevel BIC | Direction 2: Nested random change points |
|---|---|---|
| Main contribution | New derivation (theory) | New model and estimation (computation) |
| Evidence the question is open | Strong: stated as future work by Cho et al. (2024) | Moderate: no nested version found so far |
| Main risk | Derivation may not close for unbalanced designs | Slow MCMC; identifiability with short series |
| Fallback | Balanced-case theory plus simulation of conjectured penalties | Two-level model, or known-date model with random slopes |

**Preferred plan: combine both directions around one statistical question.** *In nested panel data, at which levels (ZIP code, county, region) does variation occur, both in the level of opioid supply and in its response to a policy change, and how should those levels be selected?*

1. **Core: Direction 1.** Derive the BIC penalty for three- and four-level nested models, starting from the balanced case, and evaluate it by simulation against existing criteria. This is the main theoretical contribution.
2. **Connected extension: Direction 2.** Add the 6 October 2014 rescheduling as a change point in the same multilevel model. First with the known date and a random jump at each level: this is still a linear mixed model, so the new BIC decides at which levels the policy response varies, which requires extending the derivation from random intercepts to random slopes. Then, as time allows, unknown ZIP-specific break dates modelled as nested random effects.

**Why together:** both use the same nested data, mixed-model theory and simulation framework; Direction 1 provides the selection tool and Direction 2 the substantive question, so the thesis stays one methodological theme. **Scope control:** Direction 1 is the guaranteed core, and the unknown-date model has a known-date fallback, so the thesis is complete even if the extension is not finished.

**Next steps if you agree:** read the key articles in full, reproduce the two-level result of Cho et al. (2024), confirm that the data cover 2014, and then revise the thesis outline around this theme.

*Open-question statements reflect my literature search to October 2026 and will be confirmed by forward-citation search.*

## References

Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856.

Bai, J. (2010). Common breaks in means and variances for panel data. *Journal of Econometrics*, 157(1), 78–92.

Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47–78.

Brilleman, S. L., Howe, L. D., Wolfe, R., & Tilling, K. (2017). Bayesian piecewise linear mixed models with a random change point: An application to BMI rebound in childhood. *Epidemiology*, 28(6), 827–833. https://doi.org/10.1097/EDE.0000000000000723

Cho, S.-J., Wu, H., & Naveiras, M. (2024). The effective sample size in Bayesian information criterion for level-specific fixed and random-effect selection in a two-level nested model. *British Journal of Mathematical and Statistical Psychology*, 77, 289–315. https://doi.org/10.1111/bmsp.12327

Delattre, M., Lavielle, M., & Poursat, M.-A. (2014). A note on BIC in mixed-effects models. *Electronic Journal of Statistics*, 8(1), 456–475. https://doi.org/10.1214/14-EJS890

Drug Enforcement Administration. (2014). Schedules of controlled substances: Rescheduling of hydrocodone combination products from Schedule III to Schedule II. *Federal Register*, 79(163), 49661.

Greven, S., & Kneib, T. (2010). On the behaviour of marginal and conditional AIC in linear mixed models. *Biometrika*, 97(4), 773–789. https://doi.org/10.1093/biomet/asq042

Jones, R. H. (2011). Bayesian information criterion for longitudinal and clustered data. *Statistics in Medicine*, 30, 3050–3056. https://doi.org/10.1002/sim.4323

Lorah, J., & Womack, A. (2019). Value of sample size for computation of the Bayesian information criterion (BIC) in multilevel modeling. *Behavior Research Methods*, 51, 440–450. https://doi.org/10.3758/s13428-018-1188-3

Lumsdaine, R. L., Okui, R., & Wang, W. (2023). Estimation of panel group structure models with structural breaks in group memberships and coefficients. *Journal of Econometrics*, 233(1), 45–65.

Okui, R., & Wang, W. (2021). Heterogeneous structural breaks in panel data models. *Journal of Econometrics*, 220(2), 447–473. https://doi.org/10.1016/j.jeconom.2020.04.009
