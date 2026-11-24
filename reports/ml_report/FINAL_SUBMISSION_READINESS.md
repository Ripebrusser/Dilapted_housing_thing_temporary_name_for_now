> **Superseded (2026-11-12).** This is the *pre-correction* readiness assessment that
> accompanied verdict C. Every action it lists has since been carried out; the current
> verdict and evidence are in `FINAL_READINESS_VERDICT.md`. This file is retained
> unchanged below as the record of the state that triggered the correction round.

---

# Final Submission Readiness

**Project:** Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning
**Companion documents:** `FINAL_MASTER_AUDIT.md` (evidence), `FINAL_MASTER_AUDIT_CHECKLIST.md` (item-by-item outcomes)
**Verdict:** **C — NOT READY — TECHNICAL CORRECTIONS REQUIRED**

---

## Summary judgement

The analysis is good. The submission package is not finished.

Every property that determines whether the *science* is trustworthy was verified by independent recomputation: the target reconstructs exactly from the raw census extract, the twelve unmatched districts are preserved as missing rather than imputed, the feature set is documented and leakage-free, all ten model families are correctly implemented, the grouped nested CV never lets a state cross the train/test boundary, and every reported metric reproduces to zero difference. The Phase 3 report is candid — it discloses a 0.024 pp non-significant advantage, a fold where both A models invert their ranking, and importance that is only partly stable.

What is missing is what a reader is promised *at the front of the repository*: a district ranking, a usable prediction path, slides, and status documents that describe the project as it actually is.

| Dimension | State | Readiness |
| --- | --- | --- |
| Technical readiness | Pipeline complete and correct through evaluation; no prediction path | **Incomplete** |
| Research validity | Strong: valid target, honest framing, no causal claims, limitations disclosed | **Ready** |
| Reproducibility | Verified to 0.00e+00 on metrics, OOF, refits and importance | **Ready** (one command-form caveat) |
| Code quality | 21 modules reviewed line by line; guards, no leakage, no destructive paths | **Ready** (minor smells) |
| Prediction deliverables | **Absent** | **Blocking** |
| Report readiness | Phase 2/3 strong; Phase 4 has six unapplied errata; README/scope stale | **Not ready** |
| Presentation readiness | **Absent** | **Blocking** |

---

## 1. Technical readiness

**Verified working.** Data ingestion → matching → target → features → 10 model families → grouped nested CV → evaluation → error analysis → selection, all reproducible from documented commands and all internally consistent. 66/66 contract tests pass under `py -3.14`.

**Verified absent.**

- No final prediction table, ranking, risk class or deployment threshold.
- No persisted model (`models/` is empty) and no inference entry point (`src/prediction/` has only an `__init__.py`).
- No presentation artefact.

**One environment defect.** The documented commands (`py src/...`) no longer run: `py` now resolves to Python 3.15, which has no numpy/pandas/scikit-learn. Every result in this repository was produced on Python 3.14.3 and must be replayed with `py -3.14`. This is the single most likely cause of a marker concluding "it does not run".

---

## 2. Research validity

| Question | Assessment |
| --- | --- |
| Is the outcome scientifically valid? | **Yes.** Census-computed share of dilapidated census houses, a structural-condition measure. The rejected denominator was correctly rejected (D1 `Households` = 330.8 M census houses, not 247 M households). |
| Is it what the project claims? | **Yes**, and consistently: every document states it is *not* individual homelessness. No document claims otherwise. |
| Does the design support the claims? | **Yes.** State-grouped nested CV is the right evaluation for out-of-state generalisation; random CV is reported only as a reference and is explicitly not the estimate. |
| Are causal claims avoided? | **Yes.** Coefficients and importance are labelled predictive associations; correlated blocks are flagged. |
| Is the Phase 4 null result handled honestly? | **Yes.** The mandated sentence appears verbatim, and the report also states the measured tail-bias reduction so the reader does not mistake "no improvement sufficient to meet the gates" for "no change". |
| Are limitations disclosed? | **Yes**, and unusually well: 12 limitations in Phase 3 including B's non-random 12-district exclusion and the full-sample screen. |
| Does the scope document match the analysis? | **No** — it promises a model that "ranks districts", while Phase 3 warns the model must not be used to rank the worst-off. This must be reconciled (D-02). |

---

## 3. Reproducibility

| Artefact | Reproduces | Evidence |
| --- | --- | --- |
| Target (640 values) | **exactly** | independent D2 reconstruction, 0.00e+00 |
| D5 628/12 split | **exactly** | independent name-resolution |
| Feature counts 43/77 | **exactly** | manifest and matrices |
| Phase 3 metrics (30 sets) | **exactly** | recomputed from OOF, 0.00e+00 |
| Phase 3 fold MAEs (150) | **exactly** | rebuilt folds, 0 mismatches |
| Ridge B OOF and coefficients | **bit-for-bit** | refit, α = 100 ×5 |
| SVR-B permutation importance | **exactly** | refit + permutation, 0.00e+00 |
| Phase 4 artefacts (12) | **byte-identical** | sha256 re-run |
| Phase 2 report row counts | **no** | 192/48 claimed vs 288/72 actual (D-11) |

Runtime figures are not reproducible by nature and are labelled as such.

---

## 4. Code quality

Reviewed line by line: 7 data/feature modules, 4 training modules, 7 evaluation modules, 5 test files, 3 Phase 4 modules. Reviewed at function level: 4 Phase 1 profiling scripts.

Strengths: assertion-heavy loaders that fail loudly; explicit row-count guards on every join; no zero imputation; deny-list enforced in code *and* tested; scaler fitted in-fold *and* tested; no network, no credentials, no unsafe deserialisation, no machine-specific paths.

Minor defects, none affecting results: a misleading `n_configs` docstring; a shared estimator object for the grid-free dummy; a resumable merge that silently keeps the last duplicate row; an order-dependent collinearity screen; permutation importance computed only for the selected model.

---

## 5. Prediction deliverables

**This is the principal blocker.**

The project's own scope defines the deliverable as "a validated, leakage-free model that ranks districts by severe housing deprivation". What exists is a validated *evaluation* of such a model:

- `evaluation/predictions/phase3_oof_predictions.csv` — 2,982 rows of out-of-fold evaluation predictions (not a deployment artefact);
- `evaluation/predictions/tail_oof_predictions.csv` — Phase 4 experiment predictions;
- nothing else.

To close this, the project needs, at minimum: one prediction per district from the ratified Ridge B (either fold-averaged or refit per the documented recipe), a rank column with a stated tie rule and direction, an explicit statement that ranks are unreliable in the top tail, and a schema for scoring an unseen district. None of that requires new modelling or a new hyperparameter search — it is assembly of an already-validated model.

---

## 6. Report readiness

| Document | Verdict |
| --- | --- |
| `PHASE_3_MODELLING_REPORT.md` | Submission-quality. Accurate, cautious, internally consistent, all numbers traceable. |
| `PHASE_2_DATA_PREPARATION_REPORT.md` | Strong, one factual error (row counts, D-11). |
| `PHASE_4_TAIL_REMEDY_REPORT.md` | Results correct; six errata outstanding (D-05…D-10). |
| `PHASE_4_VERIFICATION.md`, `PHASE_4_INDEPENDENT_AUDIT.md` | Accurate; their recommendations were never applied. |
| `README.md` | **Must be rewritten.** Currently says Phase 1 only, no model trained, no predictions. |
| `Admin/PROJECT_SCOPE.md` | **Must be updated.** Phase table says Phases 2–4 blocked/not started. |
| `config.yaml` | Stale and unread by any code. |

---

## 7. Presentation readiness

**No presentation exists.** No `.pptx`, `.pdf`, `.key` or slide source anywhere in the repository. Nothing can be assessed for accuracy, consistency with the report, or figure quality. This is a second blocking gap for any assessment that expects slides.

There is material to build from: 17 Phase 3 figures, comparison tables, error-by-state tables, coefficient stability, and a clean Phase 4 story (five arms, six gates, no challenger).

---

## 8. Remaining blockers

| # | Blocker | Severity | Blocks submission? |
| --- | --- | --- | --- |
| 1 | No final district predictions or ranking | HIGH | **Yes** |
| 2 | No presentation | MEDIUM | **Yes**, if slides are required |
| 3 | README / PROJECT_SCOPE / config.yaml stale | HIGH | **Yes** — a reader is actively misled |
| 4 | Phase 4 errata unapplied (D-05…D-10) | HIGH/MEDIUM | **Yes** for research integrity |
| 5 | Two required tests absent | MEDIUM | Should fix before archiving |
| 6 | Documented commands fail (`py` → 3.15) | MEDIUM | **Yes** — reproducibility claim fails in practice |
| 7 | Scope promises ranking the report forbids | MEDIUM | Must reconcile |
| 8 | No inference path / persisted model | HIGH (part of #1) | Yes, if "usable model" is claimed |
| 9 | Phase 2 row-count error; duplicated data; empty dirs | LOW | No |

No blocker requires re-training, re-featurising, changing a threshold, or revisiting a ratified decision.

---

## 9. Exact actions needed before submission

Ordered, smallest-first. Nothing here starts a new modelling phase.

1. **Rewrite `README.md`** to state: Phases 1–4 complete; Phase 6 (final predictions) not executed; 640 districts, 43 predictors (A) and 628 districts, 77 predictors (B); Ridge B primary interpretable, Elastic Net secondary, SVR-RBF benchmark; grouped CV is the estimate. Delete "No model has been trained" and "No predictions exist".
2. **Update `Admin/PROJECT_SCOPE.md`**: current phase, phase-gate table, and the ranking claim — replace "ranks districts" with the qualified statement (broad ranking supported, top-tail ranking explicitly not supported).
3. **Update or delete `config.yaml`**: set the real phase, or remove it since no code reads it.
4. **Produce the Phase 6 deliverable**: one prediction per district from ratified Ridge B, rank column (documented direction and tie rule), explicit top-tail caveat, and a written scoring recipe for an unseen district. Persist the five fold models or document the refit command.
5. **Add the two missing tests** — synthetic pinball/coverage calibration, and bootstrap reproducibility — and run the suite.
6. **Apply the Phase 4 errata**: correct 6.58 → 6.250 (three files); add binomial coverage intervals; add overall bias and the p95 calibration gap to the report; name the outcome in one sentence; record the state-clustered bootstrap as an explicit amendment to the proposal and report both interval sets.
7. **Create the presentation** from existing artefacts, with numbers copied from the CSVs, not retyped.
8. **Fix the command blocks**: `py -3.14` everywhere, add `scipy` to `requirements.txt`, and state the verified interpreter and package versions.
9. **Correct the Phase 2 report row counts** to 288 and 72.
10. **Tidy the tree**: remove the duplicate dataset copies (~49 MB), the empty directories, and `_audit/`; decide deliberately whether derived data and predictions ship in the zip or are regenerated.
11. **Re-run the full suite** one final time after all edits and record the result in the submission.

Estimated effort: one focused session. No analysis changes.

---

## 10. Is it ready to submit?

**No — not yet.**

The science is defensible and the numbers are real. Three things stand between this repository and a submission that survives scrutiny: a promised deliverable that does not exist, a front page that describes the wrong project state, and an audit trail whose corrections are still outstanding.

None of them is a research failure. All of them are finishable without touching a model.

**Verdict: C — NOT READY — TECHNICAL CORRECTIONS REQUIRED.** Completing actions 1–8 above would move the project to **B — SUBMISSION READY WITH MINOR CORRECTIONS**, because no defect found here affects the correctness of the data, the target, the features, the models, the validation or the reported results.
