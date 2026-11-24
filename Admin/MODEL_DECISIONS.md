# Model and Data Decisions

Each decision records what was decided, the evidence, the consequence and its
approval status. Nothing here is a result — no model has been fitted.

---

## D-001 — Target definition: replace the proposed formula

**Status:** proposed, **requires owner approval**
**Date:** 2026-11-10
**Phase:** 1

**Decision.** Do not use `Dilapidated_Households / Households` (D1). Use the
Census-computed share of households living in dilapidated census houses from the
HLPCA district `Total` row (D2), i.e. candidate **T2**. Keep the count-based
`D1 dilapidated ÷ D5 PCA households` (**T3**) as a robustness variant on the 628
districts where it is defined.

**Evidence.**
- Σ D1 `Households` = 330,835,767 vs India 2011 households 246,740,228 and census
  houses 330,983,625 → the column is the house stock, not households (ratio 0.9996
  to houses).
- Per-district inflation ranges 0.18–2.63× (median 1.32×), so no constant correction
  is possible.
- D1 asset columns *are* household counts (electricity 67.2%, TV 47.2%, latrine
  46.9%, LPG 28.5% of 247M households vs published 67.2/47.2/46.9/28.6%).
- T2 and T3 agree at r = 0.9994, median |diff| 0.053 pp — two independent files.
- T1 is systematically 25% low (median 3.30 vs 4.40) yet correlates 0.985, i.e. the
  error distorts levels *and* ordering.

**Consequence.** The outcome changes from a mis-specified count ratio to a
Census-computed share. The research objective (district housing deprivation) is
unchanged; only its measurement is corrected. This is a target-definition fix, not
a change of research question — and it is disclosed, not silent.

**Rejected alternatives.**
- *Divide by `Rural_Households + Urban_Households`*: identical to `Households`
  (same 330,835,767), no improvement.
- *Rescale T1 by 1/1.34*: invalid, the bias is district-specific.
- *Invent a proxy from latrine/water assets*: rejected — it would redefine
  "deprivation" without a defensible Census basis, and the true condition variable
  already exists.

---

## D-002 — District key: state-qualified names, canonicalised in one module

**Status:** adopted
**Phase:** 1

**Decision.** All joins use `canonical_key(state, district)` from
`src/data/naming.py`.

**Note on codes.** D1 `District code` and D2 `District Code` are both nationally
unique 1..640 and agree exactly (joining on code gives 640 rows with all 640
canonical names matching). D5 has no district code, so one name-based key is used
for every dataset; the code join serves as an independent cross-check, not the
primary path.

**Evidence.** Six district names occur in two states (Aurangabad, Bijapur, Bilaspur,
Hamirpur, Pratapgarh, Raigarh → 12 districts). State spellings differ across files
(Orissa/Odisha, Pondicherry/Puducherry, Delhi/NCT of Delhi).

**Consequence.** D1 ↔ D2 matches 640/640 after canonicalisation; before it, 34
districts were silently unmatched.

---

## D-003 — Exclude the slum dataset (D4) from the district model

**Status:** adopted
**Phase:** 1

**Decision.** D4 is retained for narrative/descriptive use only.

**Evidence.** 32 rows at state/UT level; no district rows. Assigning a state value
to each district would create a constant-within-state feature with no district
variation (only 31 distinct values for 640 units) and would import a different
construct (slum population share, not dilapidated housing).

**Consequence.** The model has no slum predictor. This is a documented limitation,
not an oversight.

---

## D-004 — The HLPCA file (D2) cannot supply counts

**Status:** adopted
**Phase:** 1

**Decision.** Treat D2 as a share table only. Never attempt to recover household
counts from it.

**Evidence.** `Total Number of households` = 100 in 1,908/1,908 rows; components sum
to 100 ± 0.1; the file has no count column.

**Consequence.** Count-based construction must come from D1 (numerator) + D5
(denominator). Target provenance must state that T2 is a Census-computed share.

---

## D-005 — D2's trailing column block is untrusted

**Status:** adopted
**Phase:** 1

**Decision.** Ignore D2 columns from index ~145 onward (`Contition_R_Total`,
`Residence_Good`, `Residence_Livable`, `Residence_Dilapidated`, `Residence_cum_other`,
`Residence_cum_Good`) except for the explicitly validated
`Residence_Dilapidated` used in the numerator cross-check.

**Evidence.** Duplicated labels and a header/row width mismatch; the block repeats
part of the earlier condition-of-house cross-tabulation.

---

## D-006 — Primary task is regression, not classification

**Status:** proposed
**Phase:** 1 (design), enforced from Phase 3

**Decision.** Predict the continuous share. Report ranking metrics. Binarisation
only as a secondary, clearly-labelled framing.

**Evidence.** Target spans 0.2–17.7%, right-skewed; a threshold would be arbitrary
and would discard most of the signal, while 640 rows cannot support a well-calibrated
classifier with a fine cut-off.

---

## D-007 — Validation must be grouped by state

**Status:** proposed
**Phase:** 3

**Decision.** Primary metric set comes from repeated 5-fold CV grouped by state;
random CV reported as a reference only.

**Evidence.** Districts within a state share administrative, climatic and
house-construction regimes. Random folds leak that shared context and inflate skill
relative to the realistic task of scoring an unseen state.

---

## D-008 — No imputation of the 12 unresolvable household denominators

**Status:** adopted (for Phase 2 implementation)
**Phase:** 2

**Decision.** Where a district's household count cannot be attributed (the six
duplicated names in D5), record it as missing with an explicit flag. Do not impute,
and do not borrow the other state's figure.

**Evidence.** D5 holds one district per name for those six names; the delivered
value cannot be assigned to either state without a code.

**Consequence.** Count-based features are unavailable for 12 districts; the primary
target (T2) is unaffected for all 640.

---

## D-009 — T2 adopted as the sole final target

**Status:** adopted (D-001 approved by the project owner, Phase 2)
**Phase:** 2

**Decision.** `target_deprivation_rate_pct` is the D2 `Total Number of Dilapidated`
share for each district, for all 640 districts. T1 and T3 are not used as targets
anywhere in the modelling pipeline; T3 remains only as historical evidence in the
Phase 1 audit.

**Consequence.** The modelling table's target is a Census-computed share. Every
downstream document must describe it as district-level housing-deprivation
prevalence, never as homelessness.

---

## D-010 — Two normalisation strategies for count variables

**Status:** adopted
**Phase:** 2

**Decision.** Normalise count variables in one of exactly two ways:

1. **Within-table total** — a closed composition divided by the total of its own
table (`Population`, `Total_Education`, `Total_Power_Parity`, the sum of reported
household-size categories). Valid by construction, available for all 640 districts.
2. **D5 `NO_HH`** — PCA households, Rural + Urban, for standalone household counts
   whose block has no total column in D1 (amenities, assets, water, latrine).
   Available for 628 districts.

`D1.Households`, `Rural_Households` and `Urban_Households` are never denominators
(census-house stock). Enforced by `FORBIDDEN_SOURCES` in `src/features/predictors.py`
and asserted by a test.

**Why two strategies rather than one.** Applying D5 to everything would drop 12
districts for no reason; applying within-table totals to amenities is impossible
because D1 has no amenity total. The split keeps Configuration A at full coverage
while making the D5 dependency explicit and measurable in Configuration B.

---

## D-011 — Configuration A / B, compared on identical folds

**Status:** adopted for evaluation; **adoption of B pending Phase 3**
**Phase:** 2

**Decision.**

- **A (primary, 640 districts, 43 features):** D1 socioeconomic and demographic
  shares; no D5-dependent feature.
- **B (augmented, 628 districts, 43 + 34 features):** A plus household indicators
  normalised by D5 `NO_HH`.
- A is additionally evaluated **restricted to those same 628 districts**, and both
  configurations use the **same fold objects**, so the only difference is features.

**Result so far.** On identical folds (k-NN, both schemes, all five seeds), B has
lower error: grouped MAE 2.111 → 1.855, random MAE 1.498 → 1.364.

**Explicit non-decision.** This does not adopt B. One baseline model is not enough,
and more features are not automatically better. Phase 3 must confirm the gain with
regularised models and nested CV; otherwise A is preferred for interpretability and
full 640-district coverage.

---

## D-012 — Closed compositions and reference categories

**Status:** adopted
**Phase:** 2

**Decision.** Every closed composition is expressed as shares of its own table
total and one category is dropped as the reference, chosen as the largest national
component. Dropped references: Hindus, Illiterate_Education, Age_Group_0_29,
Rs 45,000–90,000 income band, 6–8 person household size.

**Evidence.** All five blocks verified additive at ratio 1.0000. Education needed a
correction: `Literate_Education` is the subtotal of the seven attainment levels
(sum(levels)/literate = 1.0000), and including it with the levels gave 1.6878.
It was removed; levels plus `Illiterate_Education` partition the table exactly.

**Consequence.** Each block has full rank, so Phase 3 linear models are estimable
without dropping further columns. The reference choice is recorded in the manifest
checks and in the Phase 2 report.

---

## D-013 — Screening thresholds and the state-mean degeneracy

**Status:** adopted
**Phase:** 2

**Decision.** Drop features with pairwise |r| > 0.98 (4 dropped); flag any feature
with |target r| > 0.90 as possible indirect leakage (none flagged; max 0.528).
Report the state-mean baseline even where it equals the national mean.

**Evidence.** Under grouped-by-state CV every test state is unseen, so the state
mean is undefined and the code falls back to the training-fold global mean; the two
baselines are therefore numerically identical (MAE 2.440, R² −0.137). Under random
CV the state mean is strong (MAE 1.556, R² 0.528).

**Consequence.** The identical rows are not a bug and must not be "fixed" by
training on test states. They quantify how much of the target is state-level and
set the real bar for Phase 3.

---

## Phase 3 decisions (summary)

Full records are in `Admin/PHASE_3_MODEL_DECISIONS.md` (P3-D-01 … P3-D-12).
The decisions that change how results must be read:

- **P3-D-01** Grouped nested CV (5 outer, 4 inner, `GroupKFold` on states) is the
  primary evaluation; random nested CV is a labelled reference only.
- **P3-D-05** Selection rule fixed in advance: grouped mean MAE, RMSE as tie-break.
  R² is reported, never used to select.
- **P3-D-06** Configuration B improves 8 of 10 families on identical districts and
  folds, but covers 628 districts; **adoption is an open coverage decision**, not a
  settled fact.
- **P3-D-07** SVR (grouped MAE 1.7248) and Ridge (1.7490) are co-leading on B; the
  difference is inside fold noise and no significance is claimed.
- **P3-D-10** District predictions are not published as risk rankings: the top 5%
  of districts are under-predicted 97–100% of the time.

## D-024 — Final Phase 3 model selection (ratified)

**Status:** ratified by the project owner at Phase 3 close
**Phase:** 3 (closing decision)
**Detail:** `Admin/PHASE_3_MODEL_DECISIONS.md`, P3-D-13 and P3-D-14

**Decision.**

1. **Configuration B is the primary modelling configuration** — 628 districts,
   77 approved predictors (D1 plus valid D5-normalised household features).
   Configuration A remains the secondary full-coverage configuration — 640
   districts, 43 predictors. B's 12 excluded districts mean its coverage is **not**
   nationally complete and must never be described as such.
2. **Ridge on Configuration B is the primary interpretable model** (grouped MAE
   1.7490, RMSE 2.2703, R² 0.3992, Spearman 0.6311). **Elastic Net on B is the
   secondary interpretable model** (1.7476). **SVR-RBF on B is the retained
   best-performing benchmark** (1.7248). No statistical superiority among the three
   is claimed: they differ by 0.024 pp against a fold SD of ≈0.17 pp.
3. **Interpretation rule.** Ridge coefficients and permutation importance describe
   predictive associations, not causal effects. Coefficient stability across outer
   folds must be checked first: 54/77 predictors keep their sign in all five folds,
   median SD/|mean| = 0.55, and the religion shares, primary education, vehicle
   ownership and male literacy coefficients are unstable. Correlated indicators must
   be read as a block, not individually.

**Consequence.** All Phase 4 work starts from Ridge on B, with SVR-RBF as the
accuracy benchmark. No Phase 1 or Phase 2 data, target definition or completed
Phase 3 result was altered by this decision.

---

## Open decisions (after Phase 3 ratification)

1. Confirm source citations/URLs for D1, D2, D4 (`DATASET_REGISTER.md` §3).
2. **Decide how to handle extreme districts**, which the retained models compress by
   ~3.6–4.9 pp at the top of the distribution (P3-D-10). Any remedy must be evaluated
   on the same grouped folds.
3. Decide whether D7–D9 may be used, and require them to beat the ratified baselines
   to be admitted.
4. Decide whether the five B features exceeding 1.0 should be clipped, left as-is
   (current), or replaced by a D1-internal household total.
5. Decide whether any district-level output table may be published (P3-D-10).
