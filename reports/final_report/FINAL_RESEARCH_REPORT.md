# Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning

**Final Research Report**

Prepared for academic submission · November 2026
Source repository: `Samrud-Housing-Deprivation-ML`

---

## Abstract

This study predicts the **prevalence of dilapidated census houses across Indian districts** using Census 2011 data. The outcome is the Census-computed share of a district's census houses classified as *Dilapidated*, a structural-condition measure of the housing stock; it is explicitly **not** a homelessness measure. Two feature configurations were constructed from district socioeconomic profiles (43 predictors, 640 districts) and from those plus household- and housing-asset indicators (77 predictors, 628 districts). Ten model families were compared under 5-fold grouped nested cross-validation that holds out entire states, so that reported performance reflects generalisation to unseen states rather than interpolation within known ones. Configuration B improved every family, and Ridge regression with α = 100 was ratified as the primary interpretable model: **MAE 1.749 (fold mean), RMSE 2.270, R² 0.399, Spearman 0.631**, against a dummy baseline of 2.450. A benchmark SVR-RBF reached MAE 1.725, a 0.024 pp difference that is not statistically distinguishable and was not adopted. Error analysis found systematic under-prediction of the most dilapidated districts (p95 group under-predicted by 4.51 pp). Five pre-registered tail-aware remedies reduced that bias by 18–34% but all failed their registered acceptance gates, costing 9–66% more overall MAE; **no challenger was promoted**. The ratified model was then fitted on all 628 eligible districts and used to produce district predictions, a ranking and relative quintile bands, with a reusable, schema-validated inference pipeline. Two fitted values fall marginally below zero and are reported as such rather than clipped. The project's contribution is a rigorously validated, fully reproducible district-level benchmark with an unusually candid account of where the model fails.

**Keywords:** housing deprivation; district-level prediction; Census of India; grouped cross-validation; Ridge regression; quantile regression; model validation

---

## 1. Introduction

Housing condition is one of the most direct observable dimensions of deprivation. India's decennial census records, for every district, how many census houses are structurally *Dilapidated* as opposed to *Livable* or *Good*. That classification yields a district-level prevalence measure which is comparable across the country, collected by a single agency under a single definition, and available for all 640 districts of the 2011 census geography.

This report asks a narrow, answerable question: **how much of the between-district variation in that prevalence can be predicted from other district-level census indicators, and how well does such a model generalise to states it has never seen?** The question is deliberately predictive rather than explanatory. It does not ask why housing decays, which would require longitudinal and household-level data this project does not have.

Three design choices shape everything that follows. First, the unit of analysis is the district, an aggregate; no individual or household inference is possible or attempted. Second, validation holds out entire states, because districts within a state share administration, climate, material costs and reporting practice, and a model evaluated on randomly split districts is measuring interpolation, not generalisation. Third, the project reports what it fails to achieve: a model that under-predicts the worst-off districts by roughly 4.5 percentage points, and five attempts to fix that which did not survive pre-registered criteria.

## 2. Problem statement

Given a vector of district-level socioeconomic and housing-stock indicators from Census 2011, predict the share of that district's census houses classified as Dilapidated.

Formally, for district *i* with feature vector **x**ᵢ ∈ ℝᵖ, estimate

  ŷᵢ = f(**x**ᵢ),  yᵢ ∈ [0, 100] (percentage points),

where *f* is learned from *n* = 628 districts and evaluated on held-out states. The outcome has mean 5.058, median 4.40, standard deviation 3.12 and a right skew of 1.154, with 124 distinct values in [0.2, 17.7]. The skew matters: most districts sit near 3–5%, a minority exceed 10%, and the errors that matter socially are concentrated in that minority.

## 3. Research objectives

1. Construct a defensible district-level target and document why the originally proposed denominator was invalid.
2. Build leakage-free feature sets from district socioeconomic and housing-stock data.
3. Compare ten model families under validation that respects state-level clustering.
4. Quantify the cost of honest out-of-state validation relative to conventional random cross-validation.
5. Test whether tail-aware remedies can reduce systematic under-prediction of the most dilapidated districts, under gates fixed before execution.
6. Deliver a reproducible prediction artefact and an inspectable inference path.

## 4. Literature and methodological background

This project draws on four bodies of work. It does **not** claim to present a systematic review of Indian housing-deprivation literature; the framing below reflects the method literature that shaped the design and the census documentation that defines the outcome.

**Housing condition as a deprivation measure.** The Census of India's Houselisting and Housing Census records the condition of each census house as Good, Livable or Dilapidated, alongside amenities such as drinking-water source, latrine type, electricity and cooking fuel. These tables are the standard published source for district-level housing quality in India and are the source of this study's outcome and of most of its predictors (Census of India, 2011a, 2011b).

**Regularised linear models.** Ridge regression adds an L2 penalty to ordinary least squares, stabilising estimates when predictors are correlated — the situation here, where 77 socioeconomic shares overlap substantially (Hoerl & Kennard, 1970). Elastic Net combines L1 and L2 penalties and performs variable selection (Zou & Hastie, 2005). Both remain the appropriate choice when the goal is a transparent, auditable score rather than maximum accuracy.

**Non-linear ensembles and kernels.** Random forests average decorrelated trees (Breiman, 2001); gradient boosting fits additive stagewise models (Friedman, 2001); support vector regression with an RBF kernel fits a margin-based non-linear function (Cortes & Vapnik, 1995). These were included as accuracy benchmarks and as a check that the linear specification was not leaving large structure unexploited.

**Quantile regression and validation design.** Conditional quantile regression estimates a chosen quantile of the outcome distribution rather than its mean, minimising the asymmetric pinball loss (Koenker & Bassett, 1978). This is the natural tool when the failure of interest is under-prediction of the upper tail. For validation, grouped cross-validation is required whenever observations are clustered, since random splitting leaks cluster-level information between training and test (Roberts et al., 2017). For uncertainty, the paired bootstrap (Efron & Tibshirani, 1993) and binomial interval methods (Wilson, 1927; Brown, Cai & DasGupta, 2001) are used.

## 5. Research gap

Three gaps motivate the design.

1. **Prediction versus explanation.** District housing-condition studies in the Indian context are predominantly descriptive or correlational. A pre-registered, grouped-validation *predictive* benchmark for this outcome — with an explicit statement of out-of-state generalisation — is what this project contributes.
2. **Validation realism.** Work that reports random k-fold accuracy for district data overstates deployability, because neighbouring districts share unobserved state-level conditions. Quantifying that gap (here 1.140 vs 1.725 MAE) is itself a contribution.
3. **Honest tail treatment.** Reporting a single average error hides the fact that the districts of greatest policy interest are exactly the ones a mean-regression model under-predicts. This project measures that failure, attempts five remedies under gates fixed in advance, and reports that none succeeded.

No claim is made that no prior work exists on any of these points; the claim is narrower — that this combination, executed with this level of verification and disclosure, is what the project provides.

## 6. Proposed methodology

```
Raw census extracts (D1, D2, D5)
  → schema and geography reconciliation (canonical STATE||DISTRICT keys)
  → target construction and independent cross-check
  → feature construction (Configuration A: 43; Configuration B: 77)
  → leakage screen (deny-list, target-correlation, collinearity)
  → 5-fold grouped nested CV (4-fold inner) over 10 families, 76 configurations
  → evaluation: fold-mean and pooled metrics, error analysis, importance, stability
  → Phase 4: five pre-registered tail remedies, six gates
  → Phase 6: ratified Ridge B refit on all 628 districts, persisted, scored, ranked
```

Every stage is a script with a documented command, a deterministic seed, and an output file whose hash was verified across re-runs.

## 7. Dataset description

| ID | File | Shape | Role |
| --- | --- | --- | --- |
| D1 | `india-districts-census-2011.csv` | 640 × 118 | Primary predictors: household amenities, demographics, education, employment |
| D2 | `india_census_housing-hlpca-full.csv` | 1,908 × 156 | Target source; district × {Rural, Urban, Total} housing condition and amenities |
| D3 | `hlpca-colnames.csv` | 156 labels | Dictionary for D2 only |
| D4 | `Slum Population Of India.csv` | 32 × 12 | Excluded: state level; four states/UTs absent |
| D5 | `census_2011.csv` (PCA) | 1,342 × 89 | Household denominator and 34 housing/asset predictors |

D1 ↔ D2 reconciliation is complete: **640/640** districts match on canonical keys. Six district names occur in more than one state (Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh); these produce 12 districts that cannot be unambiguously matched to D5, which is why Configuration B covers 628.

D4 was excluded because a 32-row state-level table cannot enter a 640-row district model, and because four states are simply absent — not zero. D6 (2001 PCA), D7–D9 (supporting 2011 tables) and D10 (2016 geography) are registered but unused; D10's vintage does not match the 2011 geography.

## 8. Data cleaning and preprocessing

The cleaning policy is deliberately conservative, and every rule is enforced in code and covered by a test.

| Rule | Implementation |
| --- | --- |
| Missing ≠ zero | No zero-fill anywhere; Configuration B has no missing values at all; the 12 unmatched districts are excluded, not imputed |
| No clipping or winsorising | No observation was capped; five Configuration B rates exceed 1.0 by up to 6.2% due to a denominator-universe mismatch and are used as-is, disclosed |
| Target-derived columns denied | D1 `Households`, dilapidated counts and the target itself are on an explicit deny-list, asserted in tests |
| Preprocessing in-fold | Median imputation and standardisation are pipeline steps fitted inside every training fold |
| Types | Numeric parsing with explicit failure on unparseable values |

The imputer is a no-op in Configuration B (0 missing cells, 0 infinite), but is retained so the pipeline is identical to the one validated in Phase 3 and safe on new data.

## 9. Target construction

The target is the Census-computed share of dilapidated census houses:

```
target_deprivation_rate_pct = D2["Total Number of Dilapidated"]  (Rural/Urban == "Total")
```

**The rejected alternative.** The initial design divided D1's dilapidated count by D1 `Households`. That column sums to **330,835,767**, which is India's census-house stock (330,983,625), not its 246,740,228 households. The error would have inflated every rate by about 34%. It was caught in Phase 1, recorded as decision D-001, and the denominator was abandoned.

**Verification.** The stored target was reconstructed independently from the raw files with a separate implementation of the canonical naming logic: **difference 0.00e+00**. A count-based reconstruction (D1 dilapidated ÷ D5 households, 628 districts) correlates with it at **r = 0.999368** with a median difference of 0.053 pp. The residual disagreement is confined to the 12 ambiguous districts and to rounding in the published percentages.

## 10. Feature engineering

Features are district-level shares and rates, each with a documented numerator, denominator and universe.

- **Configuration A (640 × 43):** D1-derived shares — asset ownership (electricity, television, radio, scooter, car, telephone), housing quality (wall, roof, floor material shares), water and sanitation, household size distribution, education levels, employment categories, religion and caste composition, plus derived rates.
- **Configuration B (628 × 77):** A's 43 plus 34 D5-derived housing-stock and asset predictors requiring a household denominator (e.g. `tapwater_share`, `mobile_only_share`, `no_bathing_share`).

**Leakage control.** `Literate_Education` (a subtotal) was excluded because education levels divided by it equal 1.000000; the four collinearity drops (`literacy_rate_female`, `nonworker_share`, `no_bathing_share`, `open_defecation_share`) are documented with their reasons. A target-correlation screen found a maximum |r| of 0.528 (`electricity_share`) and **flagged nothing**. The screen was computed once on the full sample, which is disclosed as a mild optimism in the design.

## 11. Exploratory data analysis

The outcome is right-skewed and spatially concentrated, which determines the whole modelling strategy:

| Property | Value |
| --- | ---: |
| Districts | 640 (628 eligible for B) |
| Range | 0.2 – 17.7 pp |
| Mean / median | 5.035 / 4.40 |
| Standard deviation | 3.12 |
| Skewness | 1.154 |
| Distinct values | 124 |
| Districts ≥ p90 (9.20) | 64 |
| Districts ≥ p95 (11.20) | 35 |

The top 35 districts are heavily concentrated in Assam and West Bengal (24 of 35), which is why state-grouped validation is essential: a random split would place near-duplicate state contexts on both sides. Predictors are strongly inter-correlated, with asset-ownership shares loading on a single prosperity dimension — the reason a penalised linear model is appropriate and the reason individual coefficients must be read cautiously.

## 12. Model development

Ten families were evaluated across 76 candidate configurations:

| Family | Candidates | Notes |
| --- | ---: | --- |
| Dummy (mean) | 1 | Baseline |
| Ridge | 5 | α ∈ {1, 10, 30, 100, 300}; **α = 100 selected** |
| Elastic Net | 12 | α × l1_ratio grid |
| k-NN | 8 | k × weighting |
| Decision tree | 12 | depth × min_samples_leaf |
| Random forest | 8 | trees × depth |
| Extra trees | 8 | trees × depth |
| Gradient boosting | 8 | learning rate × depth |
| Histogram gradient boosting | 8 | learning rate × depth |
| SVR-RBF | 6 | C × γ |

Hyperparameters were selected on inner folds only; the chosen value varies across outer folds, which is the correct behaviour and is recorded in the artefacts. All fits use seed 42 and `n_jobs = 1` for determinism.

## 13. Validation methodology

```
For each of 5 outer folds (GroupKFold on state):
    train on 28 states, test on 7
    For each candidate configuration:
        For each of 4 inner folds (GroupKFold on the training states):
            fit, score on inner validation
        select the best configuration
    refit the selected configuration on all outer-training rows
    predict the held-out states
```

This yields 5 × (4 + 1) = 25 fits per configuration per scheme; 304 inner fits per outer fold across 76 configurations; 1,520 per configuration-scheme and 9,120 across the six sweeps. The sealed outer test rows never influence α, preprocessing statistics, feature selection or thresholds — a property that was later verified by corrupting outer test targets and observing **bit-identical** coefficients.

Two schemes are reported. **Grouped** (state-held-out) is the primary and only generalisation estimate. **Random** 5-fold is retained as a reference showing what leakage-prone validation would have claimed.

## 14. Results and model comparison

Grouped nested CV, fold means (n = 628 unless stated):

| Model | MAE | RMSE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: |
| **SVR-RBF (benchmark)** | **1.7248** | **2.2556** | **0.3992** | 0.6258 |
| **Ridge α = 100 (primary)** | 1.7490 | 2.2703 | 0.3992 | **0.6311** |
| Elastic Net | 1.7476 | 2.2946 | 0.3923 | 0.6444 |
| Extra trees | 1.8101 | 2.3309 | 0.3727 | 0.6289 |
| Hist. gradient boosting | 1.8413 | 2.3754 | 0.3437 | 0.5856 |
| k-NN | 1.8633 | 2.4175 | 0.3135 | 0.5695 |
| Gradient boosting | 1.8677 | 2.4338 | 0.3251 | 0.5886 |
| Random forest | 1.8708 | 2.4251 | 0.3273 | 0.6268 |
| Decision tree | 2.5055 | 3.2127 | −0.1840 | 0.3752 |
| Dummy (mean) | 2.4495 | 3.1438 | −0.1157 | — |

Ridge B pooled over all out-of-fold rows: MAE 1.7496, RMSE 2.2842, R² 0.4680, Spearman 0.6672.

Elastic Net achieves the highest rank agreement (Spearman 0.644 vs Ridge 0.631) at essentially the same MAE (1.7476 vs 1.7490), which is why it is retained as the secondary model rather than discarded; Ridge is primary because its coefficients are stable enough to interpret.

**Metric definitions.** MAE is the mean absolute error in percentage points and is the primary metric because it is in the units of the outcome and insensitive to the tail's squared influence. RMSE penalises large errors more heavily. R² is the proportion of variance explained relative to the mean; it can be negative when a model is worse than predicting the mean, which is why the decision tree's −0.184 is reported rather than clipped. Spearman's ρ measures rank agreement and is the relevant statistic for a ranking use case. **Macro F1 is not used and is not relevant here:** the target is continuous, there is no class label, and F1 would require an arbitrary discretisation whose threshold would itself be a free parameter. Where a detection framing is reported (Phase 4), it is explicitly a thresholded read-out with sensitivity and precision defined against the frozen p90/p95 cuts, not a classification model.

**Random versus grouped.** The same SVR-RBF configuration scores MAE 1.1401 under random 5-fold CV and 1.7248 under grouped CV — a 51% overstatement. Only the grouped figure is quoted as performance. This gap is itself a finding: district-level models evaluated with random splits overstate deployability substantially.

**Fold variability.** Ridge B's fold MAEs are 1.937, 1.911, 1.650, 1.732 and 1.516 (mean 1.749, sd 0.178): the result is stable across folds rather than carried by one split. The two worst-error states in the reference model are Assam (+3.26 pp under-prediction) and Sikkim (+2.75), with West Bengal +1.71 and Punjab +2.13.

## 15. Configuration A versus Configuration B

| Model | A MAE (640) | B MAE (628) | Δ | B better in |
| --- | ---: | ---: | ---: | ---: |
| Elastic Net | 2.1963 | 1.7476 | −0.449 (−20.4%) | 5/5 |
| SVR-RBF | 2.1714 | 1.7248 | −0.447 (−20.6%) | 5/5 |
| Ridge | 2.1845 | 1.7490 | −0.436 (−19.9%) | 5/5 |
| Extra trees | 2.0888 | 1.8101 | −0.279 (−13.3%) | 5/5 |
| k-NN | 2.1460 | 1.8633 | −0.283 (−13.2%) | 5/5 |
| Random forest | 2.0991 | 1.8708 | −0.228 (−10.9%) | 5/5 |
| Decision tree | 2.2528 | 2.5055 | +0.253 (+11.2%) | 2/5 |
| Dummy | 2.4495 | 2.4495 | 0 | 0/5 |

The gain is uniform across families and folds: adding 34 household-denominator predictors reduces MAE by 10–21% for every model that can use them. Configuration B is therefore ratified. Two caveats belong with it: B covers 12 fewer districts (a non-random exclusion of the ambiguous-name group), and the decision tree gets worse, which is a variance effect rather than evidence against the added features.

## 16. Ridge coefficient interpretation

Ridge on B, standardised features, α = 100, coefficients averaged over the five outer folds:

| Feature | Mean coefficient | SD | Positive in |
| --- | ---: | ---: | ---: |
| `tv_share` | −0.473 | 0.039 | 0/5 |
| `hh_size_2_share` | −0.449 | 0.026 | 0/5 |
| `electricity_share` | −0.394 | 0.031 | 0/5 |
| `radio_share` | −0.360 | 0.075 | 0/5 |
| `scooter_share` | −0.354 | 0.057 | 0/5 |
| `hh_size_3_share` | +0.313 | 0.045 | 5/5 |
| `marginal_worker_share` | −0.307 | 0.036 | 0/5 |
| `other_cooking_fuel_share` | +0.301 | 0.074 | 5/5 |

Sign consistency: **54 of 77 features** keep the same sign in all five folds; 27 are positive in all five and 27 negative in all five. The interpretation is therefore limited but not vacuous: districts with higher television, electricity and two-wheeler ownership have lower dilapidated prevalence, conditional on the rest. That is an **association between correlated prosperity indicators**, not a causal effect — these shares co-move, they are measured on the same household base, and a coefficient in a correlated block is not identified as an independent mechanism. Features whose sign flips across folds are not interpreted at all, and the Phase 3 report says so explicitly.

## 17. Permutation importance

Permutation importance was computed for the benchmark and tree models on held-out rows within each fold (refitting per fold), measuring the increase in error when a feature is shuffled.

| Feature (SVR-RBF, B) | Mean importance | Positive in |
| --- | ---: | ---: |
| `tv_share` | 0.0836 | 5/5 |
| `hh_size_2_share` | 0.0609 | 4/5 |
| `mobile_only_share` | 0.0527 | 5/5 |
| `tapwater_share` | 0.0434 | 5/5 |
| `telephone_share` | 0.0385 | 5/5 |
| `scooter_share` | 0.0384 | 4/5 |
| `other_cooking_fuel_share` | 0.0373 | 4/5 |
| `electricity_share` | 0.0307 | 4/5 |

The top ten features account for 57% of total importance.

Permutation importance was computed for the **primary** model as well, under the identical protocol (each fold's Ridge refitted on its training rows, importance measured on held-out rows, 20 shuffles, seed 42):

| Feature (Ridge, B) | Mean importance | Positive in |
| --- | ---: | ---: |
| `hh_size_2_share` | 0.1255 | 4/5 |
| `tv_share` | 0.1050 | 5/5 |
| `electricity_share` | 0.0740 | 5/5 |
| `handpump_share` | 0.0586 | 5/5 |
| `radio_share` | 0.0487 | 4/5 |
| `marginal_worker_share` | 0.0483 | 4/5 |
| `pit_latrine_share` | 0.0419 | 5/5 |
| `scooter_share` | 0.0406 | 4/5 |

**7 of the 10** most important features for Ridge are also among its 10 largest coefficients, and 7 of the top 10 for SVR-RBF overlap with Ridge's — so the same prosperity dimension dominates under three different views (coefficients, Ridge importance, SVR importance). Only 26 of 77 features are positive in at least four of five folds, which is why individual attributions are read as a block rather than as a list of mechanisms.

## 18. The extreme-district problem

Aggregate error conceals a systematic failure. Splitting the out-of-fold predictions by the observed outcome:

| Group | n | MAE | Mean actual | Mean predicted | Bias (actual − predicted) |
| --- | ---: | ---: | ---: | ---: | ---: |
| All districts | 628 | 1.750 | 5.058 | 4.966 | **+0.092** |
| ≥ p90 (9.20) | 64 | 3.648 | 11.769 | 8.175 | **+3.593** |
| ≥ p95 (11.20) | 35 | 4.527 | 13.23 | 8.72 | **+4.510** |

The model is almost unbiased overall while under-predicting the extreme group by **4.5 percentage points** — it compresses the distribution toward the mean, as variance-reducing estimators do. For any use case concerned with identifying the worst-off districts, this is the central defect, and it is not visible in the headline MAE. Mean prediction in the p95 group is 8.72 against an actual 13.23: no individual district in that group is predicted anywhere near its true value. This motivated Phase 4.

## 19. Phase 4 — tail-aware experiments

Five remedies were registered before execution, all constrained to leave the validated pipeline intact:

| Arm | Definition | Selection grid |
| --- | --- | --- |
| E0 | Ratified Phase 3 Ridge OOF, reused | — |
| E1 | Weighted Ridge, threshold q75(training), λ = 2 | α ∈ {30, 100, 300} |
| E2 | Weighted Ridge, threshold q75, λ = 4 | α ∈ {30, 100, 300} |
| E3 | Weighted Ridge, threshold q90, λ = 4 | α ∈ {30, 100, 300} |
| E4a | Linear quantile regression, τ = 0.75 | L1 α ∈ {0.001, 0.01, 0.1} |
| E4b | Linear quantile regression, τ = 0.90 | L1 α ∈ {0.001, 0.01, 0.1} |

**What happened.** Every arm moved the tail in the intended direction, and every arm paid for it:

| Arm | MAE | p95 bias | p95 bias reduction | C1 (tail) | C2 (cost) | Verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| E0 | 1.750 | +4.510 | — | — | — | reference |
| E1 | 1.915 | +3.693 | −18.1% | FAIL | FAIL (+9.5%) | not accepted |
| E2 | 1.975 | +3.435 | −23.8% | FAIL | FAIL (+12.9%) | not accepted |
| E3 | 1.985 | +3.271 | −27.5% | FAIL | FAIL (+13.5%) | not accepted |
| E4a | 1.914 | +3.109 | −31.1% | N/A | FAIL (+9.4%) | not accepted |
| E4b | 2.912 | +2.994 | −33.6% | N/A | FAIL (+66.4%) | not accepted |

The registered C1 bar was a **30%** reduction. Weighted arms reached 18–28%; the quantile arms crossed it but their MAE gates are not applicable to a conditional quantile. C4 caught E4b over-correcting: it improved Assam (+59.97 pp of summed absolute-error reduction) while making West Bengal **worse** (−23.40), with 63% of the gain in one state. C6 caught the precision collapse: flagging more districts found more extremes (sensitivity 0.54–0.69) but precision fell to 0.17–0.39 against a 0.40 floor.

**The trade, quantified.** Roughly **0.12–0.21 pp of overall MAE per 1 pp of tail bias recovered**. In other words, to remove the 4.5 pp tail bias entirely would cost on the order of 0.5–0.9 pp of overall MAE — a 30–50% degradation — which no accepted criterion would tolerate.

**Registered conclusion, verbatim:** *"No improvement was demonstrated using the tested tail-aware approaches. Limited predictor information is one possible explanation, but cannot be established from these experiments alone. Further evidence, potentially including additional independent predictors, would be required to establish an information ceiling."*

**The honest reading.** "No improvement" means no improvement *sufficient to meet the pre-registered criteria*, not no change. The tail bias fell by 0.8–1.4 pp consistently across folds, and under the registered district-level bootstrap those reductions exclude zero. The gains were real, insufficient, and expensive. Collapsing either half of that sentence would misstate the result.

**A documented deviation.** The proposal registered a district-level bootstrap; the executed analysis resampled states, on the reasoning that tail districts cluster within four states. That reasoning is defensible but it was not pre-registered, and it is reported as a deviation rather than retro-fitted to the proposal. Both procedures are now computed from the stored predictions:

| Arm | Δ p95 bias, registered district-level [95% CI] | Δ p95 bias, executed state-clustered [95% CI] |
| --- | --- | --- |
| E1 | −0.817 [−0.961, −0.682] | −0.817 [−1.077, −0.487] |
| E2 | −1.074 [−1.256, −0.901] | −1.074 [−1.455, −0.722] |
| E3 | −1.239 [−1.400, −1.053] | −1.239 [−1.508, −0.704] |
| E4a | −1.401 [−1.654, −1.144] | −1.401 [−1.695, −1.068] |
| E4b | −1.516 [−2.404, −0.553] | −1.516 [−3.782, +1.251] |

District-level intervals are 1.07–3.55× narrower. Two conclusions are method-dependent and are reported as such: E4a's overall cost interval includes zero only under state clustering (district-level: +0.165 [+0.078, +0.250]), and E4b's tail interval spans zero only under state clustering. Gates were not recomputed from either — changing the acceptance rule after seeing results is exactly what pre-registration forbids.

**Quantile calibration.** E4b promises the 90th percentile and delivers 82.5% coverage (Wilson 95% CI 79.3–85.3), short of nominal by 7.5 pp with an interval excluding 90; E4a delivers 67.0% against 75% (63.3–70.6). The miscalibration is therefore a property of the fits, not sampling noise. Subgroup coverage at p95 rests on **35 districts**: 11.4% (4.5–26.0) — a 21.5 pp-wide interval, reported as description, not as proof of failure.

**Out-of-range predictions.** E4b's out-of-fold predictions reach 29.44 against an observed maximum of 17.7. This is expected for an unconstrained linear quantile model: fitting a hyperplane above 90% of a right-skewed target places it high, and extreme feature vectors extrapolate along that direction. The solver is exact, all values are finite, and the arm reproduces byte-identically. The values were **not clipped** — clipping would conceal the extrapolation and break consistency with the pinball diagnostics.

## 20. Independent verification findings

Two independent audits were performed, the second a full repository audit with 156 check items and a defect register D-01…D-21. Their findings:

**Confirmed by recomputation, not by inspection:** the target (0.00e+00 difference), the 628/12 district split, all 30 pooled metric sets (0.00e+00), all 150 fold MAEs, the Ridge refit (bit-identical, α = 100 in all five folds), SVR permutation importance (0.00e+00), coefficient means (0.00e+00), fold identity between Phases 2 and 3, all 325 Phase 4 fits, the 30-cell gate matrix, and byte-identical regeneration of 12 Phase 4 artefacts.

**Defects found, all documentary or missing-deliverable rather than analytical:** stale front-page documents describing a Phase 1 project; a promised ranking deliverable that did not exist; an incorrect q75 threshold range (6.58–7.10 stated, 6.250–7.100 actual); two required tests absent; two nominal-risk thresholds absent; the bootstrap deviation undocumented; the outcome not explicitly named in Phase 4; and a row-count error in the Phase 2 report (192/48 stated, 288/72 actual). All were corrected in this round; none required touching a model, a fold, a threshold or a metric.

**No defect was found in:** the target's validity, the absence of leakage, the fold construction, the model implementations, the metric arithmetic, or the honesty of the reported limitations.

## 21. Final district prediction system

The ratified Ridge B model (α = 100, 77 predictors) was fitted on all 628 eligible districts, persisted with a joblib pipeline, a feature manifest and metadata recording versions, data hashes and the target definition, and used to score every eligible district.

| Output | Content |
| --- | --- |
| `final_predictions_ridge_b_628.csv` | 628 rows: district key, district, state, predicted prevalence, range flag, model id, prediction type |
| `district_ranking_ridge_b_628.csv` | Rank 1–628 by predicted prevalence, ties broken by district key, with relative quintile bands |
| `final_predictions_retrospective_628.csv` | Predictions beside observed 2011 values, marked retrospective only |

**Predictions are in-sample.** Fitting on all 628 districts and scoring them reproduces the training data; these values are **not** a performance estimate and every row is labelled `refit_in_sample`. Generalisation remains the grouped OOF result.

**Range.** Predictions span −0.636 to 13.535 pp (median 4.584, mean 5.058). The maximum is below the observed maximum of 17.7 because a regularised linear model shrinks toward the mean — the same compression documented in §18. Two districts (Chamba, −0.286; Daman, −0.636) receive negative fitted values: Ridge is unconstrained, the values are physically impossible as shares, and they are **flagged and reported as computed, not clipped**. Both rank last either way, so a floor at zero would change nothing but the reported number.

**Bands.** Five equal groups by predicted rank (126/125/126/125/126). No validated external threshold for severe district prevalence exists in these sources, so the bands are **relative ranking bands**, not risk classes; every row says so. Inventing absolute cut-points would have been unfounded.

**Inference.** `predict_districts()` loads the persisted pipeline, validates that all 77 manifest features are present, numeric, finite and in manifest order (not input order), rejects unexpected columns unless explicitly tolerated, never refits the scaler, and returns identifiers with predictions. 19 tests cover the contract, including a permuted-column test, four rejection tests, and a reproducibility test.

## 22. Limitations

1. **Historical scope.** 2011 only; no temporal validation, no applicability to later years claimed. The 2011 administrative geography (640 districts) is no longer current.
2. **Ecological design.** District aggregates cannot identify household-level risk; the ecological fallacy applies to every statement here (Robinson, 1950).
3. **Modest explanatory power.** R² ≈ 0.40 (fold mean) means roughly 60% of between-district variance is unexplained. The model ranks reasonably and predicts magnitudes only coarsely.
4. **Out-of-state generalisation is harder.** Grouped CV costs 0.58 pp MAE relative to random CV; the honest number is the higher one.
5. **Twelve districts unscorable** in Configuration B because of duplicated names — a non-random exclusion.
6. **Small, concentrated tail.** 35 districts define the top 5%, 24 in two states; tail conclusions are fragile and partly about Assam and West Bengal.
7. **Tail under-prediction persists** at ~4.5 pp; five remedies failed their gates. The model should not be used alone to allocate resources to the extreme group.
8. **Interpretation limits.** 23 of 77 coefficients change sign across folds; associations are not causal; correlated features are not independently identified.
9. **No per-district uncertainty.** Bootstrap intervals describe arm comparisons, not individual predictions.
10. **Secondary data.** Provenance recorded, not independently re-verified against censusindia.gov.in.
11. **Predictive, not diagnostic.** The work does not test why dilapidation is high, and no policy claim follows from it.

## 23. Conclusion

A transparent, penalised linear model predicts district-level prevalence of dilapidated census houses with MAE 1.75 pp and R² 0.40 under state-held-out validation — meaningfully better than a mean baseline (2.45 pp) and meaningfully worse than the 1.14 pp that a leakage-prone random split would have claimed. Ten model families were compared; the accuracy leader's advantage (0.024 pp) was too small to justify abandoning interpretability.

The substantive finding is not the headline metric but the tail: the model under-predicts the districts with the most dilapidated housing by 4.5 pp, and five carefully registered attempts to fix that either failed their acceptance criteria or failed them while degrading overall accuracy by 9–66%. That is a negative result reported as a negative result, with the trade quantified at 0.12–0.21 pp of overall MAE per pp of tail bias recovered.

The project's deliverables — verified metrics, a persisted model, district predictions, a ranking with honestly-labelled relative bands, a schema-validated inference path and a full documentation trail — are reproducible from documented commands on a pinned environment, and every reported number traces to a file in `evaluation/metrics/`.

## 24. Future work

1. **New predictors, not new models.** The tail failure is consistent with missing information (housing age, construction material, climate exposure, migration). Adding predictors requires a new pre-registered proposal, not another sweep of the same features.
2. **Longitudinal design.** Linking 2001 and 2011 district data, or later rounds, would allow change modelling; the current cross-section cannot.
3. **Hierarchical modelling.** A multilevel model with state random effects would use the clustering that grouped CV currently treats as a nuisance, and could shrink state-level estimates more efficiently.
4. **Uncertainty for individual districts.** Conformal or quantile-based prediction intervals per district would make the ranking usable for decisions with asymmetric costs.
5. **External validation.** Replication on an independent district-year with the same target definition would test whether the 0.40 R² is a property of the phenomenon or of this sample.

## 25. References

1. Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
3. Census of India (2011a). *Houselisting and Housing Census — housing condition and amenities tables*. Office of the Registrar General & Census Commissioner, India.
4. Census of India (2011b). *Primary Census Abstract*. Office of the Registrar General & Census Commissioner, India.
5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
7. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
8. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
9. Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. *Proceedings of the Royal Society A*, 186(1007), 453–461.
10. Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
11. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
12. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
13. Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351–357.
14. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
15. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

*Bibliographic details were compiled from the method literature used in this project; volume, page and year details should be checked against the originals before final submission. No claim of an exhaustive literature review is made.*

## 26. Appendix — implementation and evaluation details

### A. Reproducibility record

| Item | Value |
| --- | --- |
| Interpreter | Python 3.14.3 (invoke as `py -3.14`; `py` resolves to 3.15 and lacks the stack) |
| Libraries | pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0 |
| Seed | 42 everywhere; `n_jobs = 1` |
| Fold signature | 5 outer GroupKFold on state; 4 inner on outer-training states |
| Fit counts | 25 per configuration-scheme; 304 inner fits per outer fold; 9,120 total across six sweeps; 325 in Phase 4 |
| Verification | Metrics, folds, coefficients, importance and Phase 4 artefacts reproduce to 0.00e+00 or byte-identical |

### B. Artifact map

| Path | Contents |
| --- | --- |
| `evaluation/metrics/phase3_nested_cv_results.csv` | All 30 configuration × model metric sets, fold-mean and pooled |
| `evaluation/metrics/phase3_fold_results.csv` | Per-fold MAE, RMSE, R², Spearman |
| `evaluation/metrics/phase3_ab_comparison.csv` | Configuration A vs B, per model and fold |
| `evaluation/metrics/phase3_coefficient_stability.csv` | 77 Ridge coefficients, mean/SD/sign consistency |
| `evaluation/metrics/phase3_permutation_importance.csv` | Permutation importance, per feature and fold |
| `evaluation/metrics/tail_*.csv` | Phase 4 metrics, gates, bootstrap (both cluster units), coverage intervals |
| `evaluation/predictions/final_predictions_ridge_b_628.csv` | 628 district predictions |
| `models/final/ridge_b_metadata.json` | Model provenance, hashes, versions, target definition |

### C. Tests

99 tests across six files, all executed: Phase 1 contracts 17, Phase 2 preprocessing 19, Phase 3 modelling 21, Phase 4 leakage isolation 9, Phase 4 quantile/bootstrap 14, Phase 6 inference 19. 0 failures, 0 skips.

### D. Project documentation

`Admin/PROJECT_SCOPE.md` (scope and gates) · `Admin/MODEL_DECISIONS.md` (D-001…) · `Admin/EXPERIMENT_LOG.md` (E-01…) · `reports/ml_report/PHASE_3_MODELLING_REPORT.md` · `PHASE_4_TAIL_REMEDY_REPORT.md` (with §11 corrections) · `PHASE_6_PREDICTION_SYSTEM.md` · `FINAL_MASTER_AUDIT.md` (independent audit).
