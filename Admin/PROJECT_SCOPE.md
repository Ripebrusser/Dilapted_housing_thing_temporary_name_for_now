# Project Scope

**Project:** Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning
**Original concept name:** Predicting Homelessness Risk Using Socioeconomic Data (reframed before any modelling — see §4)
**Current phase:** Phases 1–4 complete and audited; Phase 6 (prediction system) complete; Phase 5 not started.

---

## 1. Research question

> How prevalent is dilapidated housing across Indian districts in Census 2011,
> and how much of the between-district variation in that prevalence can be
> predicted from district-level socioeconomic and housing-stock structure?

The deliverable is a validated, leakage-free model that scores and ranks
districts by **predicted prevalence of dilapidated census houses**, together
with an honest statement of how much of the variation it explains and where it
fails.

## 2. Unit of analysis

The **district**: 640 districts across 35 states/Union Territories, Census 2011.
628 of them are eligible for the ratified Configuration B.

This is not an individual or household-level study. Every statement produced by
this project is a statement about districts, and district aggregates cannot
identify household-level risk.

## 3. Outcome (target)

**District-level prevalence of dilapidated census houses**, i.e. the share of a
district's census houses that the Census classifies as *Dilapidated*:

```
target_deprivation_rate_pct
    = 100 × (census houses classified Dilapidated)
            ÷ (total census houses)
```

Source: D2 (`india_census_housing-hlpca-full.csv`), column
`Total Number of Dilapidated`, on the `Rural/Urban == "Total"` row.

"Dilapidated" is a Census of India enumerator classification of the **physical
condition of the census house** (Good / Livable / Dilapidated). It refers to
structural condition, not to tenure, affordability, overcrowding or homelessness.

**Denominator correction (decision record D-001).** The initial draft divided by
D1 `Households`, which sums to 330,835,767 — India's *census-house stock* — not
its 246,740,228 households. That formula would have inflated every rate by about
34% and was rejected. The target now comes from the Census's own computed
housing-condition share, and is cross-checked against a count-based
reconstruction (D1 dilapidated ÷ D5 households) at r = 0.999368.

*Earlier documents described the target as a share of "households". That wording
was wrong; the unit is the census house, and the corrected wording above is used
throughout the current documentation.*

## 4. Explicit non-goals

| Not in scope | Reason |
| --- | --- |
| Predicting individual homelessness | The data contain no individual records and no homelessness measure. The original concept was reframed before modelling. |
| Counts of homeless people | Not supported at district level by these files. |
| Time-series or forecasting | A single 2011 cross-section; the 2001 data do not cover the target. |
| Causal inference | Observational, cross-sectional, ecological design. Coefficients are predictive associations. |
| Live dashboard, API or website | Out of scope; the project is a research artefact with a batch prediction pipeline. |
| Absolute "risk class" labels | No validated external threshold exists; only relative ranking bands are published, and they are labelled as such. |
| District rankings treated as ground truth | Ecological fallacy: a district average says nothing about a given household. |

## 5. Approved model and validation

| Item | Ratified choice |
| --- | --- |
| Primary interpretable model | **Ridge regression, Configuration B, α = 100** |
| Secondary | Elastic Net, Configuration B |
| Accuracy benchmark | SVR-RBF, Configuration B |
| Configuration | B: 628 districts × 77 predictors; A (640 × 43) for the A/B comparison |
| Validation | 5-fold GroupKFold on **state**, 4-fold inner GroupKFold on outer-training states only |
| Primary metric | MAE (fold mean), with RMSE, R² and Spearman reported alongside |
| Reported robustness | random 5-fold CV, kept separate and never used as the generalisation estimate |

Ratified results (fold means): Ridge B MAE 1.7490, RMSE 2.2703, R² 0.3992,
Spearman 0.6311; SVR-RBF 1.7248; dummy baseline 2.4495.

Phase 4 tested five pre-registered tail-aware remedies. **No challenger was
promoted.** Ridge B remains the model, unchanged, and the tail under-prediction
of the top 5% (~4.5 pp) is a documented limitation rather than a solved problem.

## 6. Deliverables

| Deliverable | Location |
| --- | --- |
| Verified metrics and comparisons | `evaluation/metrics/` |
| Final district predictions (628) | `evaluation/predictions/final_predictions_ridge_b_628.csv` |
| District ranking with relative bands | `evaluation/predictions/district_ranking_ridge_b_628.csv` |
| Persisted model + provenance | `models/final/` |
| Reusable inference | `src/prediction/inference.py` |
| Research report | `reports/final_report/FINAL_RESEARCH_REPORT.md` |
| Presentation | `reports/presentation/` |
| Defence preparation | `reports/ml_report/VIVA_PREPARATION.md` |

## 7. Scope boundaries and known limitations

1. **Historical, single year (2011).** No temporal validation is possible. The
   results describe 2011, not the present.
2. **Small n.** 640 (or 628) districts with up to 77 predictors; capacity is
   constrained and R² ≈ 0.40 leaves most variance unexplained.
3. **Ecological design.** District aggregates cannot identify household risk.
4. **2011 administrative geography.** 640 districts, 35 states/UTs; later splits
   (e.g. Telangana, 2014) are not mapped.
5. **Twelve districts are unscorable** in Configuration B because their names are
   duplicated across states and cannot be matched to a household denominator.
   They are excluded, not imputed.
6. **Tail is small and concentrated.** 35 districts define the top 5%, 24 of them
   in Assam and West Bengal; tail conclusions are correspondingly fragile.
7. **Secondary data.** Provenance of the delivered extracts is recorded in
   `Admin/DATASET_REGISTER.md` but not independently re-verified against
   censusindia.gov.in.
8. **Slum data cannot enter the model** — D4 is state-level with four states
   absent (`Admin/DATASET_AUDIT.md` §6.3).
9. **Linear extrapolation.** The final Ridge can produce fitted values slightly
   outside [0, 100]; they are flagged, not clipped.

## 8. Phase gates

| Phase | Content | Status |
| --- | --- | --- |
| 1 | Discovery, audit, target feasibility, geography, leakage assessment | **Complete** |
| 2 | Cleaning, feature engineering, target construction, baselines | **Complete** |
| 3 | Model training, comparison, cross-validation, robustness | **Complete** |
| 3.5 | Independent audit and defect register | **Complete** (verdict C, corrections applied) |
| 4 | Extreme-district tail investigation (five registered arms) | **Complete** — no challenger promoted |
| 5 | Further modelling / new hypotheses | **Not started** — requires a new proposal |
| 6 | Final prediction system, ranking, inference | **Complete** |
| 7 | Final report, presentation, defence material | **Complete** |

Phase 5 remains deliberately closed: adding arms after observing Phase 4 results
would be the post-hoc search the pre-registration exists to prevent.
