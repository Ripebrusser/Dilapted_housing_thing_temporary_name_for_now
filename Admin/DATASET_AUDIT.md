# Dataset Audit — Phase 1

Everything in this document was produced by script, not by inspection by eye.
Reproduce with:

```bash
py src/data/audit_datasets.py      # profile + quality + metadata files
py src/data/audit_geography.py     # district/state reconciliation
py src/data/target_feasibility.py  # candidate target definitions
py src/data/resolve_denominator.py # denominator forensics
```

Outputs: `datasets/metadata/*.csv|json`, `reports/dataset_report/*.txt`.

Headline: **the proposed target formula is invalid as written.** The numerator is
sound, the denominator in the designated dataset is not a household count. A
corrected, cross-validated target is available. Details in §3.

---

## 1. Files inspected

| ID | File | Rows | Cols | Size |
| --- | --- | ---: | ---: | ---: |
| D1 | `india-districts-census-2011.csv` | 640 | 118 | 447,908 B |
| D2 | `india_census_housing-hlpca-full.csv` | 1,908 | 156 | 1,327,458 B |
| D3 | `hlpca-colnames.csv` | 145 | 2 | 6,851 B |
| D4 | `Slum Population Of India.csv` | 32 | 12 | 3,210 B |
| D5 | `census_2011.csv` | 1,342 | 89 | 658,499 B |
| D6 | `census_2001.csv` | 1,375 | 60 | 440,422 B |
| D7 | `education_2011.csv` | 61,857 | 45 | 12,247,723 B |
| D8 | `marital_status_2011.csv` | 42,660 | 27 | 5,787,304 B |
| D9 | `age_group_2011.csv` | 13,725 | 14 | 1,305,117 B |
| D10 | `geography.csv` | 677 | 10 | 34,921 B |

All ten loaded. No file was missing. No negative values anywhere. No exact
duplicate rows in D1, D2, D4, D5, D6 or D10.

## 2. Per-dataset profile

### 2.1 D1 — District census 2011 (primary)

- **640 rows = 640 districts, 35 states/UTs.** One row per district; no Rural/Urban split.
- **No missing values** (0 null cells in 640 × 118), **no duplicate rows**.
- `District code` is **nationally unique 1..640** and agrees exactly with D2's
  `District Code`: joining the two files on code yields 640 rows and all 640
  state+district names match. D5 carries no district code at all, so the canonical
  `(state, district)` key in `src/data/naming.py` remains the join path used
  everywhere, with the code join retained as a cross-check.
- Column families: identity (3), population/sex (3), literacy (3), caste (6),
  work (10), religion (8), education (10), age (4), household amenities and
  assets (≈45), house condition (1), house structure/tenure (≈4), household size
  (9), marital couples (7), income bands (12).
- **Complete column inventory (118):**

  *Identity*: District code, State name, District name.
  *Population*: Population, Male, Female.
  *Literacy*: Literate, Male_Literate, Female_Literate.
  *Caste*: SC, Male_SC, Female_SC, ST, Male_ST, Female_ST.
  *Work*: Workers, Male_Workers, Female_Workers, Main_Workers, Marginal_Workers,
  Non_Workers, Cultivator_Workers, Agricultural_Workers, Household_Workers,
  Other_Workers.
  *Religion*: Hindus, Muslims, Christians, Sikhs, Buddhists, Jains,
  Others_Religions, Religion_Not_Stated.
  *Household amenities*: LPG_or_PNG_Households, Housholds_with_Electric_Lighting,
  Households_with_Internet, Households_with_Computer.
  *Household universe*: Rural_Households, Urban_Households, **Households** (see §3).
  *Education*: Below_Primary_Education, Primary_Education, Middle_Education,
  Secondary_Education, Higher_Education, Graduate_Education, Other_Education,
  Literate_Education, Illiterate_Education, Total_Education.
  *Age*: Age_Group_0_29, Age_Group_30_49, Age_Group_50, Age not stated.
  *Assets*: Households_with_Bicycle, Households_with_Car_Jeep_Van,
  Households_with_Radio_Transistor, Households_with_Scooter_Motorcycle_Moped,
  Households_with_Telephone_Mobile_Phone_Landline_only,
  Households_with_Telephone_Mobile_Phone_Mobile_only,
  Households_with_TV_Computer_Laptop_Telephone_mobile_phone_and_Scooter_Car,
  Households_with_Television, Households_with_Telephone_Mobile_Phone,
  Households_with_Telephone_Mobile_Phone_Both.
  *Housing condition*: **Condition_of_occupied_census_houses_Dilapidated_Households**.
  *Kitchen/bathing/latrine*:
  Households_with_separate_kitchen_Cooking_inside_house,
  Having_bathing_facility_Total_Households,
  Having_latrine_facility_within_the_premises_Total_Households,
  Type_of_bathing_facility_Enclosure_without_roof_Households,
  Type_of_fuel_used_for_cooking_Any_other_Households,
  Type_of_latrine_facility_Pit_latrine_Households,
  Type_of_latrine_facility_Other_latrine_Households,
  Type_of_latrine_facility_Night_soil_disposed_into_open_drain_Households,
  Type_of_latrine_facility_Flush_pour_flush_latrine_connected_to_other_system_Households,
  Not_having_bathing_facility_within_the_premises_Total_Households,
  Not_having_latrine_facility_within_the_premises_Alternative_source_Open_Households.
  *Water*: Main_source_of_drinking_water_Un_covered_well_Households,
  Main_source_of_drinking_water_Handpump_Tubewell_Borewell_Households,
  Main_source_of_drinking_water_Spring_Households,
  Main_source_of_drinking_water_River_Canal_Households,
  Main_source_of_drinking_water_Other_sources_Households,
  Main_source_of_drinking_water_Other_sources_Spring_River_Canal_Tank_Pond_Lake_Other_sources__Households,
  Main_source_of_drinking_water_Tank_Pond_Lake_Households,
  Main_source_of_drinking_water_Tapwater_Households,
  Main_source_of_drinking_water_Tubewell_Borehole_Households,
  Location_of_drinking_water_source_Within_the_premises_Households,
  Location_of_drinking_water_source_Near_the_premises_Households,
  Location_of_drinking_water_source_Away_Households.
  *Household size*: Household_size_1_person_Households,
  Household_size_2_persons_Households, Household_size_1_to_2_persons,
  Household_size_3_persons_Households, Household_size_3_to_5_persons_Households,
  Household_size_4_persons_Households, Household_size_5_persons_Households,
  Household_size_6_8_persons_Households,
  Household_size_9_persons_and_above_Households.
  *Tenure*: Ownership_Owned_Households, Ownership_Rented_Households.
  *Married couples*: Married_couples_None_Households,
  Married_couples_1_Households, Married_couples_2_Households,
  Married_couples_3_Households, Married_couples_3_or_more_Households,
  Married_couples_4_Households, Married_couples_5__Households.
  *Income bands*: Power_Parity_Less_than_Rs_45000,
  Power_Parity_Rs_45000_90000, Power_Parity_Rs_90000_150000,
  Power_Parity_Rs_45000_150000, Power_Parity_Rs_150000_240000,
  Power_Parity_Rs_240000_330000, Power_Parity_Rs_150000_330000,
  Power_Parity_Rs_330000_425000, Power_Parity_Rs_425000_545000,
  Power_Parity_Rs_330000_545000, Power_Parity_Above_Rs_545000,
  Total_Power_Parity.

**Compositional redundancy to resolve in Phase 2:** `Household_size_1_to_2_persons`
and `Household_size_3_to_5_persons` duplicate sums of their components;
`Households_with_Telephone_Mobile_Phone` duplicates the landline/mobile/both
components; `Married_couples_3_or_more_Households` overlaps the 3/4/5+ bands;
`Total_Education` and `Total_Power_Parity` are column totals. `Households_with_
TV_Computer_Laptop_Telephone_mobile_phone_and_Scooter_Car` is a joint-ownership
indicator, not a marginal.

### 2.2 D2 — HLPCA housing abstract (primary, source of the target)

- **1,908 rows, 156 columns, 640 district keys.** Rows are district × `Rural/Urban`
  with `Total`, plus one `State - LAKSHADWEEP` row (Lakshadweep has a single district).
- Row structure per key: **628 keys have all three rows**; **9** have only
  `Total` + `Urban` (Hyderabad, Mumbai, Mumbai Suburban, Chennai, Kolkata, Central,
  New Delhi, and two Puducherry units); **3** have only `Total` + `Rural`
  (Nicobars, Kinnaur, Lahul & Spiti). Every key has a `Total` row — no imputation needed.
- **The file is percentage-normalised.** `Total Number of households` equals 100 in
  **1,908 / 1,908** rows, and Good + Livable + Dilapidated sums to 99.9–100.1 in all
  rows. Raw counts cannot be recovered. This contradicts the expectation that c11–c139
  are counts (as D3's descriptions say).
- **Damaged trailing block.** Columns from index 145 onward include duplicated labels
  (`Contition_R_Total`, `Residence_Good`, `Residence_Livable`, `Residence_Dilapidated`,
  `Residence_cum_other`, `Residence_cum_Good`) that repeat part of the earlier block.
  Treat as suspect; do not build features from the trailing block.
- D3's dictionary maps codes **c11–c145** onto this file's measurement columns. It
  does **not** describe D1 and must never be joined to D5–D9.

### 2.3 D4 — Slum population (secondary, excluded from the district model)

- **32 rows = 31 states/UTs + an `India` aggregate row** (row 0). 12 columns.
- Columns: index, `#`, `State`, `Slums` (% of state), `Slum HH`, `Slum Pop`, `Male`,
  `Female`, `Child (0-6)`, `SC`, `ST`, `Literacy`.
- Counts are quoted strings with thousands separators (`"13,920,191"`); shares are
  strings with `%` (`"5.41%"`, `"77.72 %"`). Everything numeric needs explicit parsing.
- **Geographic level is state/UT, not district.** This is a hard blocker for district use.
- Four census states/UTs have **no row at all**: Dadra & Nagar Haveli, Daman & Diu,
  Lakshadweep, Manipur. These are *not reported / not covered*, not zero.
- Exact zeros do occur (`SC`=0 in 3 rows, `ST`=0 in 5) and are plausible genuine zeros
  for small UTs — they must not be confused with non-reporting.

### 2.4 D5–D10 — optional supporting extracts

- **D5 (PCA 2011):** 1,342 rows = 634 districts × (Rural, Urban) + 36 states × 2 + 2 India.
  634 district names, each with exactly 2 rows. **No state column and no district code**,
  so a join to D1 is name-only. 170 null cells (0.14%) in 85 columns.
- **D6 (PCA 2001):** 1,375 rows; 10.7% null cells; district count (650 rural / 649 urban)
  differs from 2011 because boundaries changed. `UNNAMED: 64` column is a trailing-comma
  artefact. Not used.
- **D7/D8/D9:** C-series tables keyed by `state_code`, `dist_code`, `area_name`,
  `total`, `age_group`. 3,045 / 2,100 / 881 exact duplicate rows respectively — all
  duplicates are **also duplicates on the natural key**, i.e. redundant repeats, not
  distinct records. They carry `state_code` + `dist_code`, so they *can* be joined by
  code (unlike D5).
- **D10 (geography.csv):** 2016 vintage, 36 states, 640 districts, includes Telangana.
  Only 629 of 640 D1 districts match: the 11 unmatched are the Telangana districts
  (Adilabad, Nizamabad, Karimnagar, Medak, Hyderabad, Rangareddy, Mahbubnagar, Nalgonda,
  Warangal, Khammam) which the 2016 file assigns to Telangana, plus `New Delhi`
  (absent from the 2016 district list). **Do not use D10 as a 2011 district lookup.**

---

## 3. Target feasibility — verdict

### 3.1 Proposed definition

```
target_deprivation_rate = (Dilapidated households / Total households) × 100
numerator   = D1 Condition_of_occupied_census_houses_Dilapidated_Households
denominator = D1 Households
```

### 3.2 What is present

| Requirement | Result |
| --- | --- |
| Numerator exists | **Yes** — D1, integer, 0 nulls, range 24 … 263,800, national sum 13,202,421 |
| Denominator exists | **Yes** — D1, integer, 0 nulls, range 3,230 … 3,618,151 |
| Zero denominators | **0** districts |
| Numerator ≤ denominator | **0** violations |
| Missing targets silently zeroed | **No** — 0 nulls, nothing coerced |
| Compatible geography | **Yes** — same district rows |
| **Compatible population** | **No** — see §3.3 |

### 3.3 The denominator is not a household count

| Evidence | Value | Interpretation |
| --- | --- | --- |
| Σ D1 `Households` | 330,835,767 | — |
| India, census houses 2011 | 330,983,625 | **ratio 0.9996** |
| India, households 2011 | 246,740,228 | households are 25% fewer |
| Σ D5 `NO_HH` (PCA households) | 247,577,528 | matches households, not houses |
| Σ D1 `Population` | 1,210,854,977 | **exactly** India 2011 population |
| Σ D1 `Rural_Households` + `Urban_Households` | 330,835,767 | identical to `Households` — the same wrong universe |
| Per-district `Households` ÷ D5 households | median **1.32**, p05 1.16, p95 1.78, min 0.18, max 2.63 | not a constant rescale, so no single correction factor exists |

Cross-check that D1's *asset* columns really are household counts (national sums vs
Census 2011 published shares):

| Column | National sum | ÷ 247M households | Published 2011 |
| --- | ---: | ---: | ---: |
| Housholds_with_Electric_Lighting | 165,897,294 | **67.2%** | 67.2% |
| Households_with_Television | 116,493,624 | **47.2%** | 47.2% |
| Having_latrine_facility_within_the_premises | 115,737,458 | **46.9%** | 46.9% |
| LPG_or_PNG_Households | 70,422,883 | **28.5%** | 28.6% |

So the file mixes universes: asset columns are households, while `Households`
(and its rural/urban components) is the census-house stock. Dividing a household
count by a house count produces a ratio that is wrong by a district-specific
factor between 0.18 and 2.63.

### 3.4 Verdict

> **REJECTED as specified.** `Dilapidated_Households / Households` is not a
> defensible deprivation rate: numerator and denominator describe different
> populations. It would understate deprivation nationally (3.99% vs 5.28%) and
> distort the *ordering* of districts, because the bias varies by district.
> The target is also not homelessness and must never be described as such.

**But a valid target is feasible.** Two independent, mutually validating
constructions exist:

| ID | Definition | n | Distribution |
| --- | --- | ---: | --- |
| T1 | D1 dilapidated ÷ D1 `Households` (proposed) | 640 | median 3.30, mean 3.76, max 13.87, national 3.99 |
| **T2** | **Census-computed share of households in dilapidated census houses (D2 `Total` row)** | **640** | median 4.40, mean 5.04, max 17.70 |
| T3 | D1 dilapidated ÷ D5 PCA households | 628 | median 4.35, mean 5.01, max 17.74, national 5.28 |

- **T2 vs T3: Pearson r = 0.9994, Spearman 0.9992, median absolute difference 0.053 pp** (n = 628).
  Two entirely separate files agree, so both the numerator and a true household
  denominator are corroborated.
- **T1 vs T2: r = 0.9847 but median absolute difference 1.116 pp** — and the gap is
  systematic (T1 is 25% low), exactly what a 1.34× denominator inflation predicts.
- Reverse check: applying D2's share to D5 household counts reproduces D1's dilapidated
  count with median absolute relative error **0.86%** and r = 0.999, confirming the
  numerator is the "Total Number of Dilapidated" household count (c14), not the
  residence-only variant (c13).

**Recommended target (pending owner approval): T2.** It is the Census's own
cross-tabulation, available for all 640 districts, internally consistent
(Good + Livable + Dilapidated = 100), and independently corroborated by a
count-based construction at r = 0.999. T3 is retained as a robustness variant.

### 3.5 Caveats on the recommended target

1. T2 is delivered to one decimal place; ties are expected and metrics should not
   treat 0.1 pp differences as meaningful.
2. T2 is a share of *households*, so it inherits Census household definitions
   (institutional households excluded/included per HHA convention) that must be
   stated in the report.
3. T3 cannot be computed for 12 districts whose names occur in two states
   (Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh), because D5 has
   neither a state nor a district code. T2 avoids this entirely.
4. "Dilapidated" is an enumerator judgement of structural condition. It is a proxy
   for severe housing deprivation, **not** a homelessness measure.

---

## 4. Geographic matching report

| Relationship | Result |
| --- | --- |
| D1 districts | 640 (35 states/UTs) |
| D2 district keys (state + district) | 640 |
| **D1 ↔ D2 matched** | **640 / 640 (100%)** after canonicalising state spellings |
| D2 keys absent from D1 | 0 |
| Duplicate district names needing a state-qualified key | 6 names, 12 districts |
| D1 ↔ D5 by name | 640/640 nominal, but **628 reliable** (name-only join) |
| D1 ↔ D10 geography | 629/640 (2016 vintage, Telangana split) |

**State-name aliases found and verified** (`datasets/metadata/state_alias_map.csv`):
`ORISSA → ODISHA`, `PONDICHERRY → PUDUCHERRY`, `DELHI → NCT OF DELHI`
(plus `UTTARANCHAL → UTTARAKHAND`, not needed by these files but recorded).

**District alias:** `PONDICHERRY (D1) → PUDUCHERRY (D2/geography)`.

**Ambiguities that must never be resolved by name alone:** Aurangabad (Bihar,
Maharashtra), Bijapur (Chhattisgarh, Karnataka), Bilaspur (Himachal Pradesh,
Chhattisgarh), Hamirpur (Himachal Pradesh, Uttar Pradesh), Pratapgarh (Rajasthan,
Uttar Pradesh), Raigarh (Chhattisgarh, Maharashtra).

**Slum (D4) state matching:** no D4 state is missing from D1. Four D1 states/UTs
have no D4 row (Dadra & Nagar Haveli, Daman & Diu, Lakshadweep, Manipur) — genuinely
not reported, so they must stay missing, never 0.

The district crosswalk (`datasets/metadata/district_crosswalk.csv`) records, per
district: D1 codes, numerator, denominator, T1, D2 share, the T1−D2 gap, D5 household
count, match flags and the geography.csv id.

---

## 5. Data leakage risk assessment

### 5.1 Direct target leakage — must be excluded

| Column | Source | Why |
| --- | --- | --- |
| `Condition_of_occupied_census_houses_Dilapidated_Households` | D1 | the numerator itself |
| `Households` | D1 | the denominator (and mislabelled) |
| `Rural_Households`, `Urban_Households` | D1 | sum exactly to the denominator |
| `target_deprivation_rate_pct` | derived | the label |
| `Total Number of Dilapidated`, `Contition_T_Dilapidated`, `Residence_Dilapidated` | D2 | the label in D2 form |
| `Total Number of Good`, `Total Number of Livable`, `Contition_T_Good`, `Contition_T_Livable`, `Contition_R_Total`, `Residence_Good`, `Residence_Livable`, `Residence_cum_*`, `Total Number of Residence*` | D2 | the same condition-of-census-house cross-tabulation; Good+Livable+Dilapidated = 100, so any two of them determine the third |
| `c11`–`c22` per D3 | D2 | the condition-of-census-house block (Total / Residence / Residence-cum-other × Total / Good / Livable / Dilapidated) |

### 5.2 Denominator-reuse hazard — not leakage, but must be handled

Roughly 45 D1 columns are **household counts** (assets, amenities, water, latrine,
tenure, household size, couples). They are legitimate predictors **only after
normalising by a valid household denominator**. Using D1 `Households` for this would
inject the same 1.34× distortion into every feature, so it is excluded as a
normaliser too. Phase 2 will normalise by D5 `NO_HH` (628 districts) or by
population-based shares, and will document which is used.

### 5.3 Compositional dependencies — will bias naive importance but are not leakage

Many blocks are closed compositions (religion sums to population; income bands sum
to `Total_Power_Parity`; water sources sum to households). Feeding all components of
a closed composition to a linear model is collinear; use shares and drop one
reference category, or use a tree model and report the composition as a block.

### 5.4 Same-vintage circularity — accepted, must be disclosed

Predictors and target come from the same 2011 Census. This is a cross-sectional
association study, not a forecasting exercise. The model explains *contemporaneous*
structure; it cannot claim predictive lead time. Any operational use would need
current-year predictors.

### 5.5 What is *not* leakage

District population, literacy, caste/religion shares, workforce composition,
education levels, age structure and asset ownership are all measured independently
of the house-condition question and are legitimate predictors.

---

## 6. Data-quality findings

48 checks passed, 15 warnings, 1 failure (`datasets/metadata/data_quality_report.csv`).

| Status | Dataset | Check | Finding |
| --- | --- | --- | --- |
| **FAIL** | D4 | geographic_level_state_only | 32 state-level rows cannot be assigned to 640 districts |
| WARN | D2 | values_are_percentages_not_counts | all 1,908 rows normalised to 100; no recoverable counts |
| WARN | D3 | not_rfc4180_csv | unquoted commas in descriptions → split-on-first-comma parser required |
| WARN | D3 | header_is_duplicate_pair | both header labels identical; file is a code→description map |
| WARN | D4 | counts_and_percentages_are_text | `"13,920,191"`, `"5.41%"`, `"77.72 %"` need parsing |
| WARN | D4 | genuine_zero_vs_missing | SC=0 in 3 rows, ST=0 in 5; genuine zeros, not non-reporting |
| WARN | D4 | state_coverage_vs_census | 4 states/UTs absent (DNH, Daman & Diu, Lakshadweep, Manipur) |
| WARN | D5 | missing_values | 170 null cells / 85 columns (0.14%) |
| WARN | D6 | missing_values | 8,800 null cells / 56 columns (10.67%) |
| WARN | D6 | malformed_trailing_column | `UNNAMED: 64` from trailing commas |
| WARN | D7 | exact_duplicate_rows | 3,045 (all duplicated on the natural key → redundant) |
| WARN | D8 | exact_duplicate_rows | 2,100 (same diagnosis) |
| WARN | D9 | exact_duplicate_rows | 881 |
| WARN | D9 | missing_values | 3,743 null cells (1.95%) |
| WARN | D10 | missing_values | 1,356 null cells (20.03%) — `square_kms` absent for districts |
| WARN | D1–D2 | target_reproduced_from_housing_abstract | 1.081 pp median gap — **explained**, see §3 (denominator) |

Other verified-clean properties: no negative counts, no zero denominators, no
invalid percentages in D2 (all 0–100), no duplicate districts in D1, no district
present in one census extract but missing from the other.

---

## 7. Missing dataset requirements

| Requirement | Status |
| --- | --- |
| District-level homelessness counts/rates | **Not available** in any delivered file; the project does not attempt them |
| A valid district household denominator | Available for 628/640 districts via D5 (name-only join); **not available for 12** |
| District-level slum data | **Not available** (D4 is state level) |
| Authoritative source URLs / citations for D1, D2, D4 | **Not recorded**; owner action needed |
| Real household counts in D2 | **Not available** (percentages only) |
| District codes in D5 | **Not available**; blocks a fully reliable code-based join |

Nothing here prevents a defensible district model, but items 2 and 4 must be
resolved or disclosed before the final report.

---

## 8. Recommended final dataset combination

**Core (used):**
1. **D2 `Total` rows** → target (T2), 640 districts.
2. **D1** → socioeconomic predictors (population shares, literacy, caste/religion,
   work, education, age), 640 districts.
3. **D5** → household denominator for normalising D1 household counts (628 districts),
   and the T3 robustness target.

**Optional (only if it demonstrably helps and does not leak):**
4. **D7/D8/D9** → education level, marital status, age structure at district level,
   joined by `(state_code, dist_code)`; they add detail beyond D1's coarse categories.

**Excluded:**
5. **D4** — state level only; cannot enter a district model. Retain for descriptive
   context at state level and for the report narrative only.
6. **D6** — 2001 vintage; boundaries changed, not usable as predictors for a 2011 target.
7. **D10** — 2016 vintage with Telangana; unsafe as a 2011 lookup.
8. **D3** — dictionary for D2 only, never a join input.

**Bar to clear before adding (4):** the added block must improve out-of-fold
performance against the D1-only baseline on the *same* folds, or it is dropped.

---

## 9. Recommended ML problem formulation

| Aspect | Recommendation | Why |
| --- | --- | --- |
| Unit | District (n = 640) | only defensible level in the data |
| Task | **Supervised regression** on the deprivation share | preserves magnitude and ordering; 640 rows is enough for a small model |
| Target | T2 share, percent, 0.2–17.7 | Census-computed, validated |
| Secondary framing | Optional binary "top quintile deprivation" for interpretability | do not make it primary: the threshold is arbitrary |
| Splitting | **Grouped K-fold by state** + random K-fold as reference | tests generalisation to unseen states, which is the realistic use case |
| Resampling | Repeated grouped CV (≥5×5) with fixed seed | n is small; single-split metrics are unstable |
| Models | Regularised linear (Ridge/ElasticNet), gradient boosting (HistGradientBoosting / LightGBM if allowed), random forest, k-NN baseline | ordered from interpretable to flexible |
| Metrics | MAE and RMSE (pp), R², **Spearman ρ** (ranking matters for targeting) | a policy user ranks districts |
| Feature handling | shares/rates rather than raw counts; drop compositionally redundant columns; no target-derived columns | see §5 |
| Reporting | nested CV for any tuning; permutation importance; residual mapping | avoids optimistic tuning bias |

Logistic-style classification is explicitly *not* recommended as the primary task:
binarising a continuous, skewed share throws away information and introduces an
arbitrary cut-off.

---

## 10. Next steps for Phase 2 (after approval)

1. **Approve or reject the target change** (T2 primary, T3 robustness). This is the
   one decision that blocks everything else.
2. Build `src/preprocessing/` loaders that read the canonical keys from
   `src/data/naming.py` — never re-implement name matching.
3. Materialise `datasets/processed/district_analysis.csv`: 640 rows, district keys,
   target, predictor shares; plus `datasets/metadata/feature_manifest.csv` recording
   each feature's source, formula, denominator and leakage status.
4. Normalise D1 household counts with D5 `NO_HH`; keep an explicit missing-indicator
   for the 12 unresolvable districts rather than imputing silently.
5. Resolve composition blocks (drop redundant totals, one reference category).
6. Screen predictors for near-duplication of the target (structure/condition blocks).
7. Freeze a baseline table before any model is fitted; write it to
   `evaluation/metrics/`.
8. Only then start Phase 3 (training).
