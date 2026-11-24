# Phase 2 — Data Preparation Report

**Status:** complete, awaiting Phase 3 approval.
**Decision applied:** D-001 approved — T2 is the sole final target.
**Reproduce:**

```bash
py src/data/build_dataset.py          # builds the dataset + baselines
py tests/test_phase2_contracts.py     # 19 contract tests
```

No advanced model was trained, no tuning was performed and no final prediction
was produced in this phase.

---

## 1. Stop-condition check

| Stop condition | Result |
| --- | --- |
| Target cannot be reconstructed from D2 | **Not triggered** — 640/640 districts, 0 nulls, range 0.2–17.7% |
| Any district mapping ambiguous | **Not triggered for the modelling table** — D1↔D2 is 1:1 on both code (1–640) and canonical name. The 12 D5-ambiguous districts are excluded from Configuration B only, as authorised |
| Target leakage unresolved | **Not triggered** — deny-list enforced in code (`FORBIDDEN_SOURCES`), 0 violations, max feature–target |r| = 0.528 |
| Unexpected missing/duplicate records in the 640-district table | **Not triggered** — 640 rows, unique keys, 0 null target, 0 null in Configuration A |
| D5 denominator mapping fails validation | **Not triggered** — exactly 628 resolvable, 12 preserved as missing |

## 2. Deliverables

| File | Rows | Notes |
| --- | ---: | --- |
| `datasets/processed/district_analysis.csv` | 640 | keys + target + 43 A + 34 B features + `has_d5_households` |
| `datasets/processed/feature_manifest.csv` | 81 | every built feature, formula, denominator, source, retention |
| `evaluation/metrics/phase2_baselines.csv` | 288 | long-format fold means (4 models × 4 metrics × 6 configuration-scheme cells) |
| `evaluation/metrics/phase2_baselines_wide.csv` | 72 | configuration × scheme × model (4 models × 6 cells × 3 metric columns + keys) |
| `evaluation/metrics/phase2_folds.csv` | 1,908 | fold assignments (grouped 640, grouped 628, random rep1) |
| `evaluation/metrics/phase2_run_summary.json` | — | machine-readable run record |
| `tests/test_phase2_contracts.py` | 19 tests | all passing |

`district_analysis.csv` columns: `key`, `state`, `district`, `district_code`,
`target_deprivation_rate_pct`, `has_d5_households`, 43 A features, 34 B features.

## 3. Target

**Definition (approved, D-001):**

```
target_deprivation_rate_pct
  = D2["Total Number of Dilapidated"]  for the district's "Total" row
```

- **Source:** `datasets/raw/india_census/india_census_housing-hlpca-full.csv`
- **Source column:** `Total Number of Dilapidated` (Census code c14, per D3)
- **Subset:** `Rural/Urban == 'Total'`, one row per district
- **Unit:** percent of households; Census-computed share of the condition-of-census-house table
- **Interpretation:** district prevalence of dilapidated census houses. **Not** a homelessness measure.
- **Rejected:** T1 (`D1 dilapidated / D1 Households` — invalid denominator: census-house stock) and T3 (`D1 dilapidated / D5 households` — robustness check only, not a target).

**Distribution (n = 640):** min 0.2, p05 1.2, p25 2.7, median 4.40, mean 5.04,
p75 6.63, p95 11.2, max 17.7, sd 3.12, skew 1.15, 124 distinct values, 0 nulls,
0 negative, 0 above 100.

Validation: the three condition components sum to 100 ± 0.5 for every row
(asserted in `build_target`), so the numerator is a share of all households, not
of a subset.

The 124 distinct values reflect the one-decimal precision of the source; ties are
expected and differences under 0.1 pp must not be read as substantively meaningful.

## 4. Feature construction

81 features built, 4 dropped by screening, **77 retained**.

### 4.1 Two normalisation strategies

| Strategy | Denominator | Coverage | Used for |
| --- | --- | --- | --- |
| Within-table total | the block's own total (`Population`, `Total_Education`, `Total_Power_Parity`, sum of household-size categories) | 640 | religion, education, age, income bands, household size |
| D5 PCA households | `NO_HH` (Rural + Urban) | **628** | amenities, assets, water, latrine, tenure-free household indicators (Configuration B) |

D1's `Households` column is **never** used as a denominator; it is the
census-house stock (330.8M vs 247M actual households). That rule is enforced by
`FORBIDDEN_SOURCES` and tested.

### 4.2 Closed compositions

Each composition was checked against its own total before use:

| Block | sum(components)/total | Reference category dropped |
| --- | ---: | --- |
| religion | 1.0000 | Hindus |
| education attainment | 1.0000 | Illiterate_Education |
| age structure | 1.0000 | Age_Group_0_29 |
| income bands | 1.0000 | Rs 45,000–90,000 band |
| household size | 1.0000 | 6–8 persons |

`Literate_Education` was found to be the **subtotal** of the seven attainment
levels (sum(levels)/literate = 1.0000). Including it along with the levels gave a
1.69 ratio, so it was removed to prevent double counting; levels plus
`Illiterate_Education` partition the table exactly.

### 4.3 Configuration A — primary (640 districts, 43 features)

D1 socioeconomic and demographic shares only, no D5-dependent feature:
male share; literacy (total, male, female; female dropped by the collinearity
screen); SC and ST shares; work participation (workers, main, marginal,
non-workers dropped as the complement); cultivator / agricultural-labour /
household-industry / other-worker shares of workers; 7 religion shares; 7
education-attainment shares; 3 age shares; 7 income-band shares; 6 household-size
shares; owned-vs-rented tenure share.

### 4.4 Configuration B — augmented (628 districts, 43 + 34 features)

Adds 34 household indicators normalised by D5 `NO_HH`: LPG/PNG, electricity,
internet, computer, bicycle, car, radio, scooter, TV, telephone (total,
mobile-only, landline-only, both), multi-asset ownership; bathing facility,
no-bathing, enclosure-without-roof, latrine within premises, open defecation,
pit latrine, other latrine, night-soil open drain, flush-to-other-system,
separate kitchen, other cooking fuel; 8 drinking-water sources and 3
water-source locations.

### 4.5 Screens

**Collinearity** (|r| > 0.98, computed on the full sample and documented):
4 features dropped — `literacy_rate_female` (with `literacy_rate`),
`nonworker_share` (complement of `worker_share`), `no_bathing_share` (with
`bathing_facility_share`), `open_defecation_share` (with
`latrine_within_premises_share`).

**Indirect target leakage** (|r| > 0.90): none flagged. The strongest single
association is `electricity_share` at |r| = 0.528 — exactly the moderate
association a deprivation study should show, not a leak.

**Deny-list** (target components, D2 condition columns, D1 `Households`):
0 violations.

## 5. Missingness and coverage

- Configuration A: **0 missing values** in 640 × 43.
- Configuration B: each of the 34 features is missing for **exactly 12**
  districts — Aurangabad (Bihar, Maharashtra), Bijapur (Chhattisgarh, Karnataka),
  Bilaspur (Himachal Pradesh, Chhattisgarh), Hamirpur (Himachal Pradesh, Uttar
  Pradesh), Pratapgarh (Rajasthan, Uttar Pradesh), Raigarh (Chhattisgarh,
  Maharashtra). Preserved as missing; no zero-fill, no imputation, no borrowing
  the other state's value.
- B rates exceeding 1.0: five features, by at most 6.2%
  (`separate_kitchen_share` 1.062, `electricity_share` 1.036,
  `bathing_facility_share` 1.028, `latrine_within_premises_share` 1.023,
  `tapwater_share` 1.015). Cause: D1 asset counts and D5 household counts are
  different per-district universes. Values are **not clipped** (that would hide
  the mismatch); the bound is asserted at ≤ 1.10 in the tests.

## 6. Baselines and cross-validation

Folds: primary = 5-fold **grouped by state** (35 states; every test fold contains
states unseen in training). Reference = shuffled 5-fold, 5 seeds. Both
configurations use the **same fold objects**. All preprocessing (k-NN scaling)
is fitted inside the training fold only.

**Grouped by state (primary, honest generalisation):**

| Configuration | Model | MAE | RMSE | R² | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| A (640) | national mean | 2.440 | 3.149 | −0.137 | — |
| A (640) | state mean | 2.440 | 3.149 | −0.137 | — |
| A (640) | k-NN, 6 headline features | 2.362 | 3.078 | −0.088 | 0.358 |
| A (640) | k-NN, all 43 features | 2.124 | 2.727 | 0.074 | 0.482 |
| A (628) | k-NN, all 43 features | 2.111 | 2.704 | 0.092 | 0.452 |
| **B (628)** | **k-NN, all 77 features** | **1.855** | **2.390** | **0.329** | **0.577** |

**Random 5-fold, mean of 5 seeds (reference):**

| Configuration | Model | MAE | RMSE | R² | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| A (640) | national mean | 2.415 | 3.111 | −0.011 | — |
| A (640) | state mean | 1.556 | 2.117 | 0.528 | 0.715 |
| A (640) | k-NN, all 43 features | 1.483 | 1.935 | 0.603 | 0.748 |
| A (628) | k-NN, all 43 features | 1.498 | 1.961 | 0.597 | 0.743 |
| **B (628)** | **k-NN, all 77 features** | **1.364** | **1.837** | **0.646** | **0.785** |

### Configuration A → B on identical folds and identical districts (628)

| Scheme | MAE | R² | Spearman |
| --- | --- | --- | --- |
| Grouped by state | 2.111 → **1.855** (−12.1%) | 0.092 → **0.329** | 0.452 → **0.577** |
| Random 5-fold | 1.498 → **1.364** (−9.0%) | 0.597 → **0.646** | 0.743 → **0.785** |

## 7. Interpretation and caveats

1. **Grouped CV is much harder than random CV** (MAE 2.12 vs 1.50 for the same
   model). A large share of the apparent signal is between-state context. Phase 3
   must report the grouped number as the headline.
2. **The state-mean baseline is identical to the national mean under grouped CV**
   — by construction: every test state is unseen, so the per-state mean is
   unavailable and the code falls back to the training-fold global mean. Under
   random CV the state mean is strong (MAE 1.556, R² 0.528), which quantifies how
   much of the target is state-level. Models must beat 1.556 under random CV and
   2.440 under grouped CV to be useful.
3. **Configuration B looks better on both schemes and all five seeds, but this is
   one model (k-NN) and is not yet a decision.** The added features are
   plausibly causal-adjacent (assets, sanitation), so the gain is expected; Phase 3
   must confirm it with regularised models and a nested comparison before the
   configuration is adopted. Per instruction, B is not assumed better merely for
   having more features.
4. **k-NN is a baseline, not a result.** With 43–77 correlated features and 640
   rows, k-NN's distance metric is a weak learner; the numbers above are a floor,
   not a ceiling.
5. **The collinearity and target-correlation screens use the full sample.** They
   never touch fold assignment, and they are unsupervised screens frozen before
   modelling, but they are not recomputed per fold. Phase 3 models that select
   features by target association must do so inside folds instead.
6. **D5 normalisation changes the universe slightly** (five B features exceed 1.0
   by up to 6.2%). Interpret B coefficients as "per D5 household", not per D1
   household.
7. **D7/D8/D9 are not used.** The education, marital-status and age extracts would
   add detail, but they were optional and their inclusion must earn its place
   against these baselines in Phase 3.

## 8. What was deliberately not done

- No advanced model, no hyperparameter tuning, no feature selection by target.
- No imputation, no clipping, no synthetic rows.
- No website, API or dashboard.
- No change to raw data; every output is regenerable from the scripts.
- T1 and T3 were not used as targets anywhere in the pipeline.

## 9. Phase 3 entry point

1. Adopt the frozen `district_analysis.csv` and `feature_manifest.csv` as the only
   modelling input.
2. Fit regularised linear models (Ridge / ElasticNet) and gradient boosting
   (HistGradientBoosting, LightGBM if permitted) inside the grouped folds already
   emitted in `phase2_folds.csv`.
3. Beat the registered baselines under **both** schemes, and report the grouped
   number as primary.
4. Decide Configuration A vs B on nested-CV evidence, not on feature count.
5. Keep the deny-list and the leakage test active for every new feature.
