# Final Master Audit — Complete Project Verification

**Project:** Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning
**Audit type:** independent end-to-end technical, scientific and submission-readiness audit
**Auditor stance:** nothing accepted on the strength of a prior "complete" marker, a passing test, or a good R².
**Audit code and logs:** `_final_audit/` (isolated; no project artefact written)
**Date of audit:** 12 November 2026

---

## 1. Executive summary

**Verdict: C — NOT READY — TECHNICAL CORRECTIONS REQUIRED** (§17). The corrections are **additive and documentary**, not rework: no data defect, no leakage, and no irreproducibility was found.

What was established by direct recomputation:

| Area | Outcome |
| --- | --- |
| Target | Reconstructed independently from D2 `Total` rows for all 640 districts; **identical to 0.00e+00**. Agrees with an independent D1÷D5 reconstruction at r = 0.999368, median difference 0.053 pp. |
| Geography | D1 640/640 ↔ D2; 628 D5-resolvable; the 12 unresolved districts are exactly the six duplicated names; **no imputation** |
| Features | 81 built, 4 dropped, 77 retained = 43 (A) + 34 (B); no forbidden source reaches any predictor; `Literate_Education` double-counting correctly prevented (ratio 1.000000) |
| Models | All 10 families present; **30/30 pooled metric sets reproduce to max \|Δ\| = 0.00e+00**; **150/150 fold MAEs reproduce**; Ridge on B refits **bit-for-bit** with α = 100 in every fold |
| Validation | 5-fold outer / 4-fold inner state-grouped CV; no state ever split; A628 and B628 share districts, order and folds; Phase 2 and Phase 3 fold assignments **identical** |
| Importance | SVR-RBF permutation importance reproduces to **0.00e+00**; Ridge coefficient means and the 54/77 sign-consistency count reproduce exactly |
| Phase 4 | 325 fits reconcile (180 + 120 + 25); gates 30/30; no challenger eligible |
| Tests | 17/17, 19/19, 21/21, 9/9 — all executed by the auditor, not merely quoted |

What is **not** there:

1. **No final district predictions, rankings or risk classification exist.** `models/` is empty, `src/prediction/` contains only an `__init__.py`, `notebooks/11_final_predictions.ipynb` is a 786-byte placeholder, and `evaluation/predictions/` holds only out-of-fold evaluation files. Phase 6 was planned in `PROJECT_SCOPE.md` and never executed.
2. **No reusable inference pipeline** — nothing can score an unseen district.
3. **No presentation** — no `.pptx`, `.pdf` or slide artefact anywhere (the "PPT" cannot be audited because it does not exist).
4. **The Phase 4 audit errata were never applied.** All six (D1–D6, D13) remain in the files exactly as the earlier audit left them.
5. **Top-level status documents contradict the project's actual state**: `README.md` still says "Phase 1 complete — awaiting approval… No model has been trained… No predictions exist", `PROJECT_SCOPE.md` marks Phases 2–4 "Blocked"/"Not started", and `config.yaml` still records `phase: 1.`

Science and engineering of what exists are sound and reproducible. The submission is incomplete because a promised deliverable class is absent and the orientation documents are stale.

---

## 2. Repository inventory

176 files at the moment the audit's hash baseline was taken (excluding `__pycache__`; no `.git` and no virtual environment exist in the tree). No git repository is present.

| Area | Files | Purpose | Phase | Inspected | Status |
| --- | ---: | --- | --- | --- | --- |
| `src/data/` | 6 modules | naming, loaders, dataset build, 4 audit scripts | 1–2 | 2 full, 4 function-level | PASS / PARTIAL |
| `src/features/` | 2 modules | target, predictors + screening | 2 | full | PASS |
| `src/models/` | 2 modules | registry (10 families), tail arms | 3–4 | full | PASS |
| `src/training/` | 4 modules | splits, nested CV, runners | 3–4 | full | PASS |
| `src/evaluation/` | 7 modules | metrics, baselines, compare, error analysis, stability, finalize, tail eval | 3–4 | full | PASS |
| `src/prediction/` | `__init__.py` only | intended inference | 6 | — | **MISSING** |
| `datasets/raw/` | 10 CSVs | delivered census extracts | 1 | full (profiled) | PASS |
| `datasets/processed/` | 2 CSVs | modelling table, manifest | 2 | full | PASS |
| `datasets/metadata/` | 10 files | manifest, dictionary, crosswalk, findings | 1 | spot-checked | PASS |
| `datasets/downloads/`, `datasets/samrus (1–3)/` | 6 zips + 9 CSVs | **duplicate** copies of the raw data | 1 | hashed | REDUNDANT |
| `evaluation/metrics/` | 24 files | Phase 2/3/4 metrics, params, logs | 2–4 | full | PASS |
| `evaluation/predictions/` | 4 CSVs | OOF predictions only | 2–4 | full | PASS (no final predictions) |
| `evaluation/plots/` | 17 PNGs | Phase 3 figures | 3 | existence + code | PASS |
| `evaluation/error_analysis/` | 6 CSVs | residual tables | 3 | full | PASS |
| `models/` (incl. `final/`) | 0 files | persisted models | 6 | — | **EMPTY** |
| `notebooks/` | 11 | 01 + 06–10 executed; 02–05, 11 placeholders | 1–4 | structure + outputs | PARTIAL |
| `tests/` | 5 | 66 contract tests + notebook runner | 1–4 | full | PASS |
| `reports/` | 15 md/txt | dataset, Phase 2/3/4, audit, proposal | 1–4 | full / partial | PASS |
| `Admin/` | 7 md | scope, register, audit, methodology, decisions, log | 1–4 | partial | **STALE in places** |
| `Docs/`, `reports/final_report/`, `datasets/interim/` | 0 files | intended destinations | — | — | **EMPTY** |
| `_audit/` | 3 files | previous Phase 4 audit scripts | 4 | — | stray audit output at root |

**Excluded from the source audit:** `__pycache__/`, `*.pyc`, `.freebuff/`, binary `.zip` payloads (listed, hashed), `.png` plot binaries (existence and generating code verified, pixels not audited).

**Explicit redundancy.** `datasets/raw/` (22 MB) duplicates `datasets/samrus (1–3)/` (20 MB + 1.8 MB), and `datasets/downloads/` (7.3 MB) duplicates `datasets/samrus (1).zip`. Spot-hashed pairs (`census_2011.csv`, `education_2011.csv`, `india-districts-census-2011.csv`) are byte-identical. ≈49 MB of duplication in a submission package.

**Missing or orphaned items.** No model artefacts; no final prediction table; no ranking; no inference entry point; `Docs/` and `reports/final_report/` empty; `config.yaml` is **never read by any code** (no `yaml` import anywhere) and carries a stale phase marker; `notebooks/11_final_predictions.ipynb` is an empty placeholder for a deliverable that was never built.

---

## 3. Phase 1 — dataset audit

Independent profiling, re-reading the raw CSVs with the auditor's own naming logic:

| Dataset | Expected | Measured | Status |
| --- | --- | --- | --- |
| D1 district census | 640 × 118 | **640 × 118**, 35 states, unique keys, no duplicate rows | PASS |
| D2 HLPCA | 1,908 × 156 | **1,908 × 156**, 3 Rural/Urban levels, 640 `Total` rows, keys unique | PASS |
| D3 dictionary | 156 columns | 156-entry dictionary for D2 only | PASS |
| D4 slum | 32 × 12, state-level | 32 rows, state-level | PASS (excluded from model) |
| D5 PCA 2011 | 1,342 × 89 | **1,342 × 89**; 1,268 district rows, 634 distinct names, every name exactly 2 rows (rural+urban) | PASS |

**Target validity.** `target_deprivation_rate_pct` = D2 `Total Number of Dilapidated` on the `Rural/Urban == "Total"` row.

- The `Total` row is the correct additive row: Good + Livable + Dilapidated sums to 99.9–100.1 for **every** district.
- D1's `Households` column is indeed the **census-house stock**, not households: sum = **330,835,767** (India census houses 330,983,625), versus 246,740,228 actual households. The rejected T1 denominator was correctly rejected.
- T2 agrees with an independent count-based reconstruction (D1 dilapidated households ÷ D5 households, 628 districts) at **r = 0.999368**, median absolute difference **0.0530 pp** — the README's claim of r ≈ 0.9994 / 0.053 pp is accurate.
- Distribution: 0.2–17.7, mean 5.0348, median 4.40, sd 3.12, skew 1.154, 124 distinct values, 0 nulls. Matches the Phase 2 report exactly.
- No target leakage: the target's source columns appear in no feature formula, and target-derived columns are deny-listed and tested.

**Geography.** D1↔D2 is 1:1 on canonical `(state, district)` keys — 640/640, 0 unmatched. Six district names occur in two states (Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh). D5 has no state column, so the matching rule resolves a district only when its name is unique in D1: **628 resolvable, 12 preserved as missing**, and those 12 are exactly the duplicate-name districts. Nothing is imputed, borrowed or zero-filled.

---

## 4. Phase 2 — cleaning and preprocessing

| Check | Result |
| --- | --- |
| Canonicalisation | Upper-case, punctuation-stripped, verified alias map (Orissa→Odisha, Pondicherry→Puducherry, Delhi→NCT of Delhi, Uttaranchal→Uttarakhand); aliases are additive, never a silent district rename | PASS |
| Join key | `state||district`; district names alone are not used for joins | PASS |
| Duplicate handling | 0 duplicates in D1, D2 `Total`, processed table; 4 near-duplicate *features* dropped by screen | PASS |
| Missing values | Configuration A: 0. Configuration B: exactly 12, exactly the same 12 rows, for all 34 features | PASS |
| No fabricated denominators | The 12 unresolved districts are `NaN`, **never 0** (tested and independently confirmed) | PASS |
| Invalid percentages | Target bounded 0–100 and asserted; features bounded and asserted (see overshoot note) | PASS |
| Total-row extraction | D2 filtered to `Total` before any arithmetic; prefix `District - ` stripped | PASS |
| Denominator consistency | Within-table totals for closed compositions; D5 `NO_HH` (rural+urban) only for household indicators; D1 `Households` never used | PASS |
| Rural/urban aggregation | D5 summed over the two district rows (every name has exactly 2) | PASS |
| Outlier handling | None — no clipping, capping or winsorising (correct: the target is the phenomenon of interest) | PASS |
| Zero-fill | None anywhere | PASS |
| Reproducibility | Deterministic; rebuilt values match the stored table exactly | PASS |
| Meaning-changing operations | Five B rates exceed 1.0 by ≤ 6.2% (`separate_kitchen_share` 1.0618, `electricity_share` 1.0363, `bathing_facility_share` 1.0283, `latrine_within_premises_share` 1.0225, `tapwater_share` 1.0153) because D1 counts and D5 households are different universes. **Disclosed** in the Phase 2 and Phase 3 reports, not clipped | PASS (disclosed) |

**Screening limitation (the item the brief asks to investigate).** The collinearity screen (`|r| > 0.98`) is computed **once on the full sample, before any fold split**, and frozen. It dropped 4 features: `literacy_rate_female`, `nonworker_share`, `no_bathing_share`, `open_defecation_share`. The target-correlation screen (`|r| > 0.90`) is also computed on the full sample; its maximum observed association is `electricity_share` at **|r| = 0.528** and **nothing was flagged**, so no feature was kept or dropped on the basis of the target.

Consequences, stated precisely: the *feature set* is not chosen inside folds, so it is not fold-adaptive; but it is transductive in the weak sense that unsupervised correlations used all 640 rows including future held-out folds. Because no label-driven decision changed, the effect on the reported CV estimates is negligible-to-small, and it is disclosed in `METHODOLOGY.md` (§126, §148), the Phase 2 report (§5, limitation 5) and the Phase 3 report (limitation 9). The residual risk is that a different sample would have dropped a different 4 features; the direction of any bias is not established.

---

## 5. Phase 3 — EDA and feature engineering

- **81 features built, 4 dropped, 77 retained.** Configuration A = **43**, Configuration B = **43 + 34 = 77**. Both match the manifest and the model matrices exactly (no undocumented predictor, no documented-but-absent feature).
- Every retained feature has a formula, denominator, denominator source, universe and leakage status recorded in `feature_manifest.csv`; all are marked `clear`.
- **Education correction verified.** `Literate_Education` is the subtotal of the seven attainment levels: sum(levels) ÷ `Literate_Education` = **1.000000**. It is excluded; levels + `Illiterate_Education` partition `Total_Education` at **1.000000**. Seven education shares are retained (the reference category, `Illiterate_Education`, is dropped by the compositional convention).
- Compositions are additive by construction (religion 1.0000, education 1.0000, age 1.0000, income 1.0000, household size 1.0000), each with one reference category dropped.
- **No target-derived predictor, no post-outcome information, no duplicate feature name, no identifier column** in any model matrix (asserted and independently re-checked).
- Feature names are identical between training and inference because both read the frozen manifest.
- EDA notebooks `03_eda.ipynb` and `04_feature_engineering.ipynb` are **placeholders**; the EDA record that exists is in the Phase 2 report and `Admin/DATASET_AUDIT.md`, not in a notebook.

---

## 6. Phase 4 (brief) — machine-learning implementation

The brief's "Phase 4" corresponds to the repository's **Phase 3**. Mapping is stated explicitly because the numbering does not match (§13).

| Claim | Independent finding |
| --- | --- |
| Ten families | All present and registered: dummy, ridge, elastic net, kNN, decision tree, random forest, extra trees, gradient boosting, HistGradientBoosting, SVR-RBF |
| Parameters | Grids of 1/5/12/8/12/8/8/8/8/6 = **76 candidates**; arithmetic reproduces the report's 304 inner fits per outer fold, 1,520 per configuration-scheme, 9,120 across six sweeps |
| Scaling / imputation | `SimpleImputer(median)` + `StandardScaler` inside a `Pipeline`, fitted per training fold; a test asserts learned scaler means equal training-fold means and differ from full-sample means |
| Seeds | 42 throughout; `n_jobs=1`; HistGB `early_stopping=False`; deterministic across runs |
| Target/feature input | Target column and identifiers never enter X (asserted) |
| Metrics | 30 pooled metric sets recomputed from OOF: **max \|Δ\| = 0.00e+00**. 150 fold-level MAEs recomputed from independently rebuilt folds: **0 mismatches** |
| Reference table in the brief | Matches the **fold-mean** (`mae_mean`) figures exactly: A640 SVR 2.0293/2.6144/0.1699/0.4626; A628 ExtraTrees 2.0888/2.6540/0.1837/0.4966; B628 SVR 1.7248/2.2556/0.3992/0.6258. Their **pooled** counterparts differ (B628 SVR pooled R² 0.4755) — the Phase 3 report labels both, so the table is correct but must be cited with its definition |
| Primary configuration | B (628 d, 77 f) — confirmed by the ratified decision records and the report |
| Ridge primary interpretable | Confirmed: MAE 1.7490, α = 100 in all five folds, coefficients interpretable |
| Elastic Net secondary / SVR benchmark | Confirmed: Elastic Net 1.7476 (R² 0.3923), SVR-RBF 1.7248 (R² 0.3992, best MAE) |
| Dummy baseline | 2.4401665 in Phase 3 vs 2.4402 in Phase 2 — reproduces the Phase 2 mean baseline, confirming harness consistency |
| Model persistence | **None exists** (`models/` empty). Nothing to verify; nothing can be reloaded |

---

## 7. Phase 5 (brief) — model validation and robustness

| Check | Result |
| --- | --- |
| Outer folds | 5-fold `GroupKFold` on 35 states; 512/128 for the 640 configuration, 502–503/125–126 for the 628 configurations — matches the report |
| Inner folds | 4-fold, carved from outer-training rows only (tested) |
| Grouping | No state in both sides of any fold — verified directly on all 5 folds for both configurations |
| A/B fold identity | A628 and B628 share districts, row order, targets, groups **and identical fold index arrays**; Phase 2 and Phase 3 fold assignments are **identical for all 640 districts** |
| Random CV | Reported separately, never used for selection (21/21 tests) |
| Preprocessing leakage | None: pipelines fitted in-fold; scaler test passes |
| Target leakage | None: deny-list enforced, 0 violations, max feature–target \|r\| = 0.528 |
| Hyperparameter selection | Args from inner folds only; per-fold choices recorded and differing across folds, as reported |
| OOF generation | One prediction per district per model; 40 groups of 628 and 20 of 640 rows; no duplicates, no nulls, finite |
| Aggregation | Fold means and pooled metrics both reproduced from OOF |
| Error analysis | Residuals, state errors, target bands, extremes, largest errors — all computed from OOF only (`error_analysis.py` consumes OOF exclusively) |
| Permutation importance | Recomputed for B628/SVR-RBF: **max \|Δ\| = 0.00e+00**; measured on held-out rows with per-fold refit |
| Coefficient stability | Recomputed: means **Δ = 0.00e+00**, 54/77 sign-consistent reproduced |
| Causal claims | None. The report states repeatedly that these are predictive associations; correlated compositional blocks are explicitly flagged as destabilising importance |
| Comparability | Model comparisons use identical districts and folds wherever required |

**Two substantive caveats, both disclosed in the report and both correct:**

1. Only **6/43** (A640), **6/43** (A628) and **10/77** (B) features are positive in all five folds. The report says so and warns that top-ranked features with 2/5 folds positive are not reliable.
2. Fold 5 inverts ranking for both A models (ρ ≈ −0.05, R² < 0), which the report states rather than averaging away.

**One structural observation not stated in the report:** permutation importance exists **only for the selected model per configuration** (A640/SVR, A628/ExtraTrees, B628/SVR). **Ridge — the ratified primary interpretable model — has no permutation importance at all**; its interpretability rests on coefficient stability (§12.1), which is defensible but should be stated explicitly so a reader does not look for a Ridge importance table.

---

## 8. Phase 6 (brief) — final predictions and district outputs

**This phase does not exist in the repository.**

| Requirement | Finding |
| --- | --- |
| District prediction table | **ABSENT.** No file maps 628/640 districts to a production prediction. `evaluation/predictions/` contains only OOF evaluation files (`phase3_oof_predictions.csv`, `tail_oof_predictions.csv`, and two `_phase3_*` intermediates). |
| District ranking | **ABSENT.** No ranking file, no rank column, no sorted deliverable |
| Risk classification | **ABSENT.** No threshold-based class, no documented class boundaries — correct in the sense that no unvalidated classifier was invented, but the promised artefact is missing |
| Identifier / mapping correctness | Only testable on OOF: keys unique, 628/640 coverage exact, actual values equal the processed target, no district duplicated or dropped |
| Units | Percentage points throughout; predictions finite and within plausible range (OOF) |
| Actual/predicted swap | Not present — columns are explicit and reconstructed consistently |
| Reusable inference pipeline | **ABSENT.** `src/prediction/__init__.py` is empty; there is no `fit`/`predict` entry point, no serialised model, no schema for an unseen district |
| Regenerability | OOF is regenerable (verified). A *final* prediction file is not regenerable because it was never produced |

**Consequence.** The project's stated deliverable in `PROJECT_SCOPE.md` §1 — "a validated, leakage-free model that ranks districts by severe housing deprivation" — is not delivered. What exists is a validated *evaluation* of such a model. The Phase 3 report's limitation 6 additionally warns the model "must not be used to rank the worst-off districts", which is compatible with ranking *broadly* but directly contradicts the unqualified "ranks districts" claim in the scope document. That contradiction must be resolved in the final documentation, not silently.

---

## 9. Code quality audit

Line-by-line for the 21 modules listed as "full" in §2; function-level for the four Phase 1 profiling scripts.

| File | Functions inspected | Defects found | Tests performed | Status |
| --- | --- | --- | --- | --- |
| `src/data/naming.py` | `normalise_name`, `canonical_state`, `canonical_district`, `canonical_key`, `strip_area_prefix` | none | alias/prefix tests; 640 unique keys; 6 duplicates | PASS |
| `src/data/loaders.py` | `_read`, `assert_unique`, `load_d1`, `load_d2_total`, `load_d5_households`, `assert_merge` | none; row-count and uniqueness guards are strong | rebuild matches stored table exactly; 628/12 split reproduced | PASS |
| `src/data/build_dataset.py` | `check`, `main` | none functionally; writes processed + baselines in one pass | full independent recomputation of target/features/counts | PASS |
| `src/data/audit_datasets.py` | `add_quality`, `classify_role`, `profile_columns`, `read_dataset`, `raw_row_widths`, `main` | not line-by-line inspected (33 KB profiler) | outputs re-derived independently from raw data | PARTIAL |
| `src/data/audit_geography.py` | `load`, `main` | not line-by-line inspected | 640/640 matching re-derived independently | PARTIAL |
| `src/data/resolve_denominator.py` | `main` | not line-by-line inspected | conclusion (D1 `Households` = house stock) re-derived independently: 330.8 M | PARTIAL |
| `src/data/target_feasibility.py` | `main` | not line-by-line inspected | T2 vs T3 agreement re-derived independently: r = 0.999368 | PARTIAL |
| `src/features/target.py` | `build_target`, `target_metadata`, `describe_target` | none; asserts additivity, range, uniqueness | target identical to independent rebuild | PASS |
| `src/features/predictors.py` | `_arr`, `_share`, `build_features`, `composition`, `screen_features`, `screen` | Full-sample screening (documented, disclosed); screen is **order-dependent** by construction (`candidates[:index]`) so the retained set depends on declaration order — deterministic, but a reordering would change the 4 drops | 43/77 counts, deny-list, education ratio, missingness pattern | PASS (documented) |
| `src/models/registry.py` | `ModelSpec`, `n_configs`, `_pipeline`, `build_models`, `describe_models` | `n_configs` docstring says "grid crossed with the inner CV" but the property returns the **grid size only** — misleading comment, no effect on results (the summary column is labelled "candidates" and equals the grid size) | 76 candidates recomputed; 10 families | PASS (minor) |
| `src/models/tail_aware.py` | full module | none (audited previously and re-checked) | 180 inner thresholds, 25 refits, corruption tests | PASS |
| `src/training/nested_cv.py` | `build_outer_folds`, `_inner_splitter`, `run_model`, `run_configuration` | For grid-free specs the **shared** `spec.estimator` is fitted in place (only `dummy_mean`); correct per fold but a shared-state smell. `record.update(regression_metrics(...))` imports inside the loop | full OOF reproduction; dummy reproduces Phase 2 | PASS (minor) |
| `src/training/splits.py` | `grouped_folds`, `random_folds`, `assert_folds_valid`, `fold_table` | none | fold validity, coverage, group integrity | PASS |
| `src/training/run_nested_cv.py` | `build_configurations`, `_merge_csv`, `main` | Resumable merge keeps the **last** row on duplicate keys, so a partial rerun silently supersedes earlier results (documented as resumable, but there is no provenance of which run produced which row) | config shapes 640×43, 628×43, 628×77 | PASS (informational) |
| `src/training/run_tail_experiments.py` | `preflight`, `main` | none | 325-row audit trail reconciles | PASS |
| `src/evaluation/metrics.py` | `summarise_folds`, `pooled_metrics`, `metric_table` | none | fold/pooled reproduction | PASS |
| `src/evaluation/baselines.py` | `regression_metrics`, `predict_*`, `run_baselines`, `summarise` | `state_mean` falls back to the global mean for unseen states — intended and documented | baseline MAE 2.4402 reproduces | PASS |
| `src/evaluation/compare.py` | `load_phase2_reference`, `build_comparison`, `ab_paired_comparison`, `fold_stability`, `selection_summary` | `rank(method="first")` breaks ties arbitrarily (only affects display) | selection = argmin grouped MAE in all 3 configurations | PASS |
| `src/evaluation/error_analysis.py` | all tables + 7 plot functions | none; OOF-only contract honoured | importance reproduction to 0.00e+00 | PASS |
| `src/evaluation/finalize_phase3.py` | `main` | Computes permutation importance for the **selected** model only — a design choice not stated in the report | deliverable files present and consistent | PASS (minor) |
| `src/evaluation/coefficient_stability.py` | `main` | none | coefficients and 54/77 sign count reproduce | PASS |
| `src/evaluation/tail_evaluation.py` | full module | Bootstrap cluster unit deviates from pre-registration (see §15 D-05) | 12/12 artefacts byte-identical; intervals reproduced | PASS (with deviation) |
| `tests/*.py` | see §10 | `run_notebooks.py` uses `exec`/`eval` on local notebook sources — no network, no external input | 66 tests executed | PASS |
| `notebooks/06–10` | structure + outputs | executed, no error outputs, execution counts present | validated by test and by inspection | PASS |
| `notebooks/02–05, 11` | placeholders | declared placeholders; 11 is a placeholder for a **missing deliverable** | — | PARTIAL |

**Data safety.** No module writes to `datasets/raw/`. No destructive operation anywhere. No mutation of a shared frame across models (X is read-only; `data[columns]` copies). No execution-order dependency: every script derives paths from `Path(__file__).resolve()` and imports its inputs explicitly. `finalize_phase3.py` depends on `_phase3_*` intermediates existing — documented in its docstring.

**Security.** No credentials, keys or tokens. No network calls (`requests`/`urllib`/`http` absent). No `pickle`/`joblib` deserialisation. No `subprocess`. `exec`/`eval` appear only in `tests/run_notebooks.py`, executing the project's own local notebooks. No machine-specific paths in code or config (verified by grep).

**Reproducibility.** Seeds fixed (42); versions recorded in `_phase3_*_meta.json` and the Phase 3 report; output schemas stable; execution order documented in the Phase 3 report's "Reproduce" block. One gap: `scipy` is used (`importlib.metadata.version`) but not pinned in `requirements.txt`, and `matplotlib` is pinned to a version not verifiable here.

---

## 10. Test suite audit

Environment: `py` now resolves to **Python 3.15** (no numpy/pandas/sklearn); all work was therefore run with **`py -3.14`** → Python 3.14.3, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1.

| Suite | Command | Passed | Failed | Skipped | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase 1 contracts | `py -3.14 tests/test_phase1_contracts.py` | **17** | 0 | 0 | 0 |
| Phase 2 contracts | `py -3.14 tests/test_phase2_contracts.py` | **19** | 0 | 0 | 0 |
| Phase 3 contracts | `py -3.14 tests/test_phase3_contracts.py` | **21** | 0 | 0 | 0 |
| Phase 4 isolation | `py -3.14 tests/test_phase4_isolation.py` | **9** | 0 | 0 | 0 |
| **Total** | | **66** | 0 | 0 | 0 |

These are substantively strong, not smoke tests:

- `test_d1_households_is_not_the_household_universe` asserts the rejected denominator's identity numerically (330.98 M ±1%).
- `test_target_is_reproduced_from_d2` compares the shipped target to a fresh reconstruction.
- `test_no_zero_imputation_of_denominators` asserts the 12 unresolved districts are `NaN` and never 0.
- `test_scaling_is_fitted_on_the_training_fold_only` asserts learned means equal training-fold means **and differ from** full-sample means — it fails if someone moves the scaler out of the pipeline.
- `test_fold_metrics_correspond_to_saved_fold_assignments` recomputes fold MAE from OOF under a rebuilt fold assignment.
- `test_ridge_result_is_reproducible` re-runs a model end-to-end to 1e-9.
- `test_notebooks_are_valid_and_contain_no_error_outputs` requires execution counts and forbids error outputs.

**Gaps that remain** (the suite cannot catch these):

1. **No synthetic test of pinball loss or quantile coverage** against a known-calibration case, although the Phase 4 proposal requires one. The implementation is correct (independently verified), but unguarded.
2. **No bootstrap reproducibility test** (also required by the proposal).
3. **No test that the executed bootstrap uses the registered cluster unit** — which is how the deviation in §15 D-05 went unnoticed.
4. Tests validate against stored artefacts, so a regenerated artefact is never cross-checked against an independent recomputation at test time (the auditor performed that role manually).
5. `test_notebooks_...` checks for error outputs but not that notebook numbers equal the CSV numbers.

No test silently skips; none is dead code; none only checks file existence.

---

## 11. Reproducibility audit

Recomputed independently and compared. Tolerances: exact equality (`1e-12`) for values that must be deterministic; `1e-9` for metric arithmetic.

| Quantity | Reported | Recomputed | Difference | Status |
| --- | --- | --- | --- | --- |
| Target (640 values) | stored table | rebuilt from D2 `Total` | **0.00e+00** | MATCH |
| D5 resolvable count | 628 | 628, same 12 missing | 0 | MATCH |
| Feature counts | 43 / 34 / 77 | 43 / 34 / 77 | 0 | MATCH |
| Education additivity | 1.0000 | 1.000000 (levels÷literate), 1.000000 (levels+illiterate÷total) | <1e-6 | MATCH |
| Max feature–target \|r\| | 0.528 | 0.528 (`electricity_share`), 0 flagged | 0 | MATCH |
| Phase 3 pooled metrics (30 sets) | CSV | from OOF | **0.00e+00** | MATCH |
| Phase 3 fold MAEs (150) | CSV | from OOF + rebuilt folds | **0** mismatches | MATCH |
| Dummy A640 MAE | 2.4402 (Phase 2) | 2.4401665 (Phase 3) | 3e-5 (rounding) | MATCH |
| Ridge B OOF | CSV | refit per fold, α=100 | **0.00e+00** | MATCH |
| Ridge coefficient means | CSV | refit | **0.00e+00** | MATCH |
| Sign-consistent predictors | 54/77 | 54/77 | 0 | MATCH |
| SVR-B permutation importance | CSV | refit + permutation (10 repeats, seed 42) | **0.00e+00** | MATCH |
| Phase 4 OOF / gates | 3,140 rows / 30 gate rows | unchanged, chain intact | — | MATCH |
| Phase 2 baselines row count | **192** (report) | **288** (file) | 96 rows | **MISMATCH** (D-11) |
| Phase 2 baselines wide row count | **48** (report) | **72** (file) | 24 rows | **MISMATCH** (D-11) |
| Phase 4 runtime total | 11.7 s (report) | 8.652 s (stored CSV) | timing only | NOT REPRODUCIBLE (D-12) |
| Phase 4 q75 threshold range | 6.58–7.10 | **6.250–7.100** | wrong lower bound | **MISMATCH** (D-06) |

**Method-level reconciliation.** The brief's Phase 3 table (2.0293 / 2.0888 / 1.7248) matches the **fold-mean** metrics. Pooled values for the same models are 2.0293 / 2.0890 / 1.7250 with R² 0.2895 / 0.2721 / 0.4755. The Phase 3 report labels both columns; Phase 4 uses pooled metrics (E0 R² 0.468). Any document mixing the two without a label is inconsistent — this audit found the Phase 3 report correct and no internal inconsistency, but the definitions must travel with every quoted number.

---

## 12. End-to-end pipeline audit

| # | Stage | Input | Transformation | Output | Next stage consumes it? | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Raw ingestion | 10 CSVs in `datasets/raw` | profiled, validated | `datasets/metadata/*` | yes | PASS |
| 2 | Geographic matching | D1, D2, D5 | canonical keys, 640/640, 628/12 | keys in memory | yes | PASS |
| 3 | Target construction | D2 `Total` rows | column select + assertions | target vector | yes | PASS |
| 4 | Cleaning / canonicalisation | D1, D2, D5 | naming, types, prefixes | clean frames | yes | PASS |
| 5 | Feature engineering | D1, D5 | 81 built, 4 dropped | `district_analysis.csv`, `feature_manifest.csv` | yes | PASS |
| 6 | Feature selection | features + target | frozen collinearity screen | manifest `retained` | yes | PASS (full-sample, disclosed) |
| 7 | Model training | `district_analysis.csv` | 10 families × grids | in-memory estimators | yes | PASS |
| 8 | Nested CV | X, y, groups | 5×4 grouped (+ random reference) | `_phase3_*` intermediates | yes | PASS |
| 9 | Evaluation | OOF predictions | metrics, folds, comparison | `phase3_*` tables | yes | PASS |
| 10 | Error analysis / plots | OOF | residuals, importance, figures | `evaluation/error_analysis/*`, `plots/*` | reporting only | PASS |
| 11 | Model selection / ratification | comparison tables | argmin MAE + decision records | `phase3_selected_models.csv`, `phase3_best_parameters.json` | yes | PASS |
| 12 | **Final predictions** | — | — | **nothing** | — | **MISSING** |
| 13 | **District ranking** | — | — | **nothing** | — | **MISSING** |
| 14 | **Model persistence / inference** | — | — | **nothing** | — | **MISSING** |

**Can the project be executed from documented commands in order?** Yes for Phases 1–4: `py src/data/audit_datasets.py` → `audit_geography.py` → `target_feasibility.py` → `resolve_denominator.py` → `py src/data/build_dataset.py` → `py src/training/run_nested_cv.py --scheme grouped` → `--scheme random` → `py src/evaluation/finalize_phase3.py` → `py src/evaluation/coefficient_stability.py` → Phase 4 runner/evaluation. Each is documented in a README or report block and takes relative paths only.

**Caveat that breaks it today:** the documented command form is `py <script>`, and `py` now resolves to Python 3.15 without the scientific stack. Every documented command fails as written unless `py -3.14` is used. Nothing in the repository pins or checks the interpreter.

**Stages 12–14 do not exist**, so the chain terminates at evaluation. There is no documented or undocumented step that produces a deployable prediction.

---

## 13. Report and presentation audit

| Document | Consistency with artefacts | Finding |
| --- | --- | --- |
| `PHASE_3_MODELLING_REPORT.md` | Title correct; target correctly described; 640/628/43/77 counts correct; 10 families and 76 candidates correct; metrics match CSVs; fold-mean vs pooled explicitly distinguished; limitations disclosed (12 items); no causal claim; explicitly "not homelessness" | **PASS** — the strongest document in the repository |
| `PHASE_2_DATA_PREPARATION_REPORT.md` | Target, 81/4/77 features, screening, education correction, 628/12 all correct; **row-count claims 192 and 48 are wrong (files hold 288 and 72)** | PASS with one factual error (D-11) |
| `PHASE_4_TAIL_REMEDY_REPORT.md` | Results match CSVs; carries the mandated null wording verbatim; **contains the uncorrected 6.58 threshold range, no binomial interval, no outcome name, no overall-bias/p95-gap rows** | PASS on results, FAIL on errata (D-06…D-09) |
| `PHASE_4_VERIFICATION.md` | Reproducibility hash table correct; repeats the 6.58 error | PASS with D-06 |
| `PHASE_4_INDEPENDENT_AUDIT.md` / `PHASE_4_AUDIT_CHECKLIST.md` | Accurate at the time of writing; **their recommended errata were never applied** | PASS (self-consistent, superseded by inaction) |
| `README.md` | "Phase 1 complete — awaiting approval", "No model has been trained", "No predictions exist", "notebooks 02–11 placeholders" | **FAIL — materially stale and misleading** (D-01) |
| `Admin/PROJECT_SCOPE.md` | "Current phase: Phase 1… awaiting approval"; Phase 2 "Blocked", Phases 3–4 "Not started"; "No model has been trained… No prediction has been produced"; deliverable promises "ranks districts" | **FAIL — materially stale**; also contradicts the Phase 3 limitation on ranking the worst-off (D-01, D-02) |
| `config.yaml` | `phase: 1`, `status: awaiting-approval`; never read by any code | **FAIL — stale and dead** (D-03) |
| `Admin/EXPERIMENT_LOG.md` | E-01…E-22 present and accurate except the 6.58 range | PASS with D-06 |
| `Admin/METHODOLOGY.md`, `MODEL_DECISIONS.md`, `PHASE_3_MODEL_DECISIONS.md` | Consistent with code; full-sample screening disclosed; ratified roles correct | PASS |
| **Presentation** | **No `.pptx`, `.pdf`, `.key` or slide source exists** | **ABSENT — cannot be audited** |

**Cross-document contradictions found:** (a) scope promises district ranking, Phase 3 forbids ranking the worst-off; (b) README says no model exists, every Phase 3/4 document reports trained models; (c) scope marks Phase 2 blocked, `datasets/processed/` contains Phase 2 deliverables. All are stale-status defects rather than analysis defects.

**Unsupported claims:** none found in the Phase 2/3/4 reports. Every numerical statement checked (target distribution, feature counts, model counts, metrics, fold sizes, baseline comparisons, importance values, coefficient values, gate outcomes) traces to a stored artefact. The Phase 3 report's caution — that SVR's 0.024 pp advantage is not established — is exemplary rather than overstated.

**References.** No bibliography, no cited literature, and therefore no unsupported attribution. Nothing to verify against external sources; also nothing to support the framing academically.

---

## 14. Repository integrity audit

- **No version control.** `git status` → "not a git repository". No history, no tags, no branches, no diff to review.
- **Hash baseline taken before any audit action:** `_final_audit/hashes_before.txt`, 176 files. Re-hashed after the audit (see `FINAL_SUBMISSION_READINESS.md`); no project file changed.
- **Phase boundaries (timestamp evidence):** Phase 1/2 outputs 15:05–15:39; Phase 3 metrics 16:02–16:34; Phase 4 `tail_*` 17:00–17:01. No Phase 3 artefact postdates Phase 4, and no Phase 2 artefact postdates Phase 3.
- **Decision records untouched** since 16:33.
- **Phase 5/6/7:** no source, metric, prediction or report exists for any of them. There is no unapproved later modelling.
- **`_audit/`** (three files) is previous-audit scratch left in the repository root — harmless, but it is not a project artefact and should not ship.
- **Dead config:** `config.yaml` is unreferenced by code.
- **`.gitignore`** excludes `datasets/processed/`, `evaluation/predictions/*.csv`, `evaluation/plots/*.png` and `models/` — i.e. the Phase 2 table, all predictions and all plots are untracked by design. In a git-hosted submission these would be absent; in a zip they are present. Either is defensible, but the delivery mechanism must be chosen deliberately.

**Immutability cannot be proven.** Without a repository, a pre-audit manifest and an external reference copy, a careful edit that preserved timestamps would be undetectable. Timestamps are mutually consistent and no Phase 4 code path writes outside `tail_*`, but that is corroboration, not proof.

---

## 15. Defect register

| ID | Severity | File/location | Finding | Evidence | Impact | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
| **D-01** | **HIGH** | `README.md`; `Admin/PROJECT_SCOPE.md`; `config.yaml` | Status documents state Phase 1 only, "no model trained", "no predictions exist", Phases 2–4 blocked/not started, `phase: 1` | grep: README lines 6–7; SCOPE lines 5, 76–80; config lines 8–9 | A marker or reader receives a document that contradicts the entire delivered project; undermines credibility of everything else | Rewrite status sections to reflect Phases 1–4 complete and Phase 6 not executed |
| **D-02** | MEDIUM | `Admin/PROJECT_SCOPE.md` §1 vs Phase 3 report limitation 6 | Scope promises a model that "ranks districts by severe housing deprivation"; Phase 3 says it "must not be used to rank the worst-off districts" | Scope §1; Phase 3 §14.6 | Contradictory claims about the same deliverable | State the qualified position once: ranks broadly (ρ 0.63), unreliable in the top tail |
| **D-03** | LOW | `config.yaml` | Never read by any code; stale `phase`, `status` | no `yaml` import in any module | Dead configuration implies a pipeline contract that does not exist | Delete or wire it up and update fields |
| **D-04** | **HIGH** | `evaluation/predictions/`, `models/`, `src/prediction/`, `notebooks/11_final_predictions.ipynb` | **No final predictions, rankings, risk classes, persisted model or inference path** | directory listings: 4 OOF CSVs only; `src/prediction/` has only `__init__.py`; notebook 11 is 786 B | The stated deliverable (a district ranking) does not exist; the project stops at evaluation | Produce a final prediction + ranking artefact from the ratified Ridge B, with documented thresholds and an explicit "not for the top tail" caveat; persist the fold models or document a refit recipe |
| **D-05** | **HIGH (record) / MEDIUM (effect)** | `src/evaluation/tail_evaluation.py`; `reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md` | Bootstrap executed as **state-clustered**; the pre-registration specifies **district-level**; the proposal still reads "district-level" and still carries the matching caveat | proposal line 163; report §4.3 | Intervals are 2–3× wider; two report statements (E4a cost interval includes zero; E4b tail gain unstable) hold only under the executed clustering. Direction is conservative, so no conclusion is inflated | Amend the proposal record explicitly and report both interval sets |
| **D-06** | MEDIUM | `PHASE_4_TAIL_REMEDY_REPORT.md` §2; `PHASE_4_VERIFICATION.md` §4; `EXPERIMENT_LOG.md` E-22 | q75 threshold range stated as **6.58–7.10**; actual **6.250–7.100** (fold 3 = 6.250) | `tail_fit_diagnostics.csv`, `tail_selected_params.json` | Factual error repeated in three documents; no effect on results | Correct to 6.250–7.100 |
| **D-07** | MEDIUM | `tests/` | The pinball/coverage synthetic-calibration test and the bootstrap reproducibility test required by the proposal **do not exist** | grep: 0 matches for pinball/bootstrap in `test_phase4_isolation.py` | Verified behaviours are unguarded; a refactor could break them silently | Add both tests (the auditor's independent checks can serve as the specification) |
| **D-08** | LOW | `tail_quantile_metrics.csv`; Phase 4 report | No **binomial interval** for quantile coverage, required "with a binomial interval that acknowledges n = 35" | file columns contain no CI | Coverage reported without uncertainty exactly where n is smallest | Add Wilson/Jeffreys intervals for overall and subgroup coverage |
| **D-09** | LOW | Phase 4 report tables | Omits overall signed bias and the p95 mean-predicted-vs-actual calibration gap, both required and both present in the CSVs | `tail_experiments.csv` has `all_bias`, `p95_mean_predicted` | Reader cannot see that every arm over-predicts the bulk (E0 +0.09 → E4b −2.29) or that even the best arm predicts 10.24 where actual is 13.23 | Add the two rows/columns |
| **D-10** | LOW | Phase 4 report | Never names the outcome (dilapidated census houses) | grep "dilapidated" → 0 | A Phase-4-only reader cannot tell what is predicted | Add one sentence |
| **D-11** | LOW | `reports/dataset_report/PHASE_2_DATA_PREPARATION_REPORT.md` §2 | Claims 192 baseline rows and 48 wide rows; files contain **288** and **72** | row counts | Report undercounts its own deliverables (6 schemes × 3 configs × 4 baselines × 4 metrics) | Correct to 288 / 72 |
| **D-12** | INFORMATIONAL | Phase 4 report §2 | Runtime "11.7 s" not recoverable; stored CSV sums to 8.652 s | `tail_runtime.csv` | Timing only; data artefacts byte-identical | Note that runtime is not a reproducible quantity |
| **D-13** | MEDIUM | `requirements.txt`; environment | Documented commands use `py`, which now resolves to **Python 3.15** without numpy/pandas/sklearn; `scipy` is used but not pinned | `py -0p`; run failure reproduced; `version("scipy")` in code | No documented command runs as written on this machine | Pin the interpreter (`py -3.14`) in every command block and add scipy to requirements |
| **D-14** | MEDIUM | `reports/`, repository root | **No presentation artefact of any kind** | `find` for ppt/pptx/pdf/key → none | Reviewers expecting slides have nothing to assess | Produce slides from the Phase 3/4 artefacts |
| **D-15** | LOW | `datasets/` | ≈49 MB of duplicated raw data (`samrus (1–3)`, `downloads`, `raw`) | hashes identical on spot-checked pairs | Bloated delivery; risk of the wrong copy being edited | Keep `datasets/raw` only; delete the duplicate trees |
| **D-16** | LOW | `src/models/registry.py` | `n_configs` docstring claims "grid crossed with the inner CV"; returns grid size only | code | Misleading comment for maintainers; no effect on results | Correct the docstring |
| **D-17** | LOW | `src/training/nested_cv.py`, `run_nested_cv.py` | Grid-free spec fitted in place (shared object); resumable merge silently keeps the last row on duplicate keys | code | No effect on stored results; weak provenance if a sweep is interrupted | Clone before fitting; record run provenance per row |
| **D-18** | INFORMATIONAL | `src/features/predictors.py` | Collinearity screen is order-dependent and full-sample | code + `phase2_run_summary.json` (4 drops) | Deterministic today; a reordering changes the retained set. Disclosed in three documents | Record the declared order as part of the frozen screen |
| **D-19** | INFORMATIONAL | `evaluation/plots/`, `notebooks/10` | 17 PNGs exist; `notebooks/10_error_analysis.ipynb` is 825 KB with embedded outputs | file sizes | Large binary payloads in a text repository | Acceptable, but exclude from any text-only submission |
| **D-20** | INFORMATIONAL | `Docs/`, `reports/final_report/`, `datasets/interim/`; `_audit/` | Empty directories advertised in the layout; previous-audit scratch at root | listings | Slightly confused structure at delivery | Remove empty dirs and `_audit/`, or populate them deliberately |
| **D-21** | LOW | `src/evaluation/finalize_phase3.py` | Permutation importance computed only for the selected model; **Ridge has none** | `phase3_permutation_importance.csv` models = {svr_rbf, extra_trees} | A reader looking for the primary model's importance finds nothing; the report should say so explicitly | State it, or compute importance for Ridge too |

**Severity totals:** CRITICAL 0 · HIGH 3 (D-01, D-04, D-05) · MEDIUM 6 (D-02, D-06, D-07, D-13, D-14, D-05-effect) · LOW 9 · INFORMATIONAL 4.

No defect affects the correctness of the target, the features, the models, the validation or the reported metrics.

---

## 16. Limitations of this audit

1. **Immutability is not proven** — no git history and no pre-existing hash manifest; only a before/after hash comparison taken by this audit.
2. **Four Phase 1 profiling scripts were inspected at function level, not line by line.** Their conclusions were independently re-derived from the raw data, so a defect in their reporting would not have propagated into this verdict, but a dormant bug in them cannot be excluded.
3. **Only Ridge (B) was refitted end-to-end**; SVR, Extra Trees and the remaining seven families were verified through their OOF predictions, fold tables and permutation importance rather than by re-running their full grids. Budget was deliberately not spent re-running ~9,120 fits.
4. **Notebook internals were not line-audited.** They were checked for validity, execution, and absence of error outputs.
5. **Metric reproduction cannot detect a wrong-but-consistent definition.** Thresholds, group sizes and fold counts were independently re-derived, which limits but does not eliminate this risk.
6. **No external literature or census-source verification.** Provenance of the delivered extracts is taken as recorded in `DATASET_REGISTER.md`.
7. **No visual inspection of the 17 PNGs.** Existence and generating code were verified; whether each figure is legible and correctly labelled was not assessed pixel by pixel.
8. **Phase 6 cannot be audited because it does not exist.** All statements about it are about absence, not correctness.
9. **This audit does not claim the project is error-free.** It claims that the specific properties tested above hold, and it names what was not tested.

---

## 17. Final verdict

# C — NOT READY — TECHNICAL CORRECTIONS REQUIRED

The core scientific work is **sound, honest and reproducible**: the target is valid and independently reconstructible, the features are correctly constructed and leakage-free, the ten implementations are correct, the nested grouped CV is properly isolated, every reported metric recomputes to zero difference, and the Phase 3 report under-claims rather than over-claims.

It is not submission-ready because:

1. a promised deliverable class — **final district predictions, rankings and a usable inference path** — was never produced (HIGH);
2. **three top-level orientation documents actively contradict the project's state**, describing a Phase 1 project with no trained model (HIGH);
3. the **Phase 4 pre-registration deviation is still unrecorded** in the proposal, and its six audit errata remain unapplied (HIGH/MEDIUM);
4. there is **no presentation**, which cannot be assessed at all (MEDIUM);
5. two required tests are missing and the documented commands no longer run in this environment (MEDIUM).

None of these requires re-doing the modelling, re-deriving a feature, or changing a ratified decision. They require producing the missing artefacts and correcting the record.
