"""Predictor construction for Phase 2.

Two normalisation strategies are used, both explicit and recorded per feature:

1. **Within-table totals** (valid by construction, available for all 640
   districts). A closed composition is divided by the total of its own table,
   e.g. income bands / ``Total_Power_Parity``. No external denominator needed.
2. **D5 PCA household counts** (``NO_HH``), used for standalone household counts
   whose block has no total column in D1 (amenities, assets, water, latrine).
   Available for the 628 districts with a reliable name match.

D1's ``Households`` column is never used as a denominator (it is the census-house
stock, see Admin/DATASET_AUDIT.md §3). Target components are never used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFIG_A = "A"
CONFIG_B = "B_only"

# Six headline features for the k-NN baseline (scaled in-fold).
KNN_FEATURES = [
    "literacy_rate",
    "sc_share",
    "st_share",
    "worker_share",
    "cultivator_share_of_workers",
    "marginal_worker_share",
]

# Never acceptable as a predictor source, in any configuration.
FORBIDDEN_SOURCES = {
    "Condition_of_occupied_census_houses_Dilapidated_Households",
    "Households",
    "Rural_Households",
    "Urban_Households",
    "Total Number of Dilapidated",
    "Total Number of Good",
    "Total Number of Livable",
    "Contition_T_Dilapidated",
    "Contition_T_Good",
    "Contition_T_Livable",
    "Contition_T_Total",
    "Contition_R_Total",
    "Residence_Dilapidated",
    "Residence_Good",
    "Residence_Livable",
    "Residence_cum_other",
    "Residence_cum_Good",
}

RELIGION = [
    ("hindu_share", "Hindus"),
    ("muslim_share", "Muslims"),
    ("christian_share", "Christians"),
    ("sikh_share", "Sikhs"),
    ("buddhist_share", "Buddhists"),
    ("jain_share", "Jains"),
    ("other_religion_share", "Others_Religions"),
    ("religion_not_stated_share", "Religion_Not_Stated"),
]

EDUCATION = [
    ("below_primary_share", "Below_Primary_Education"),
    ("primary_share", "Primary_Education"),
    ("middle_share", "Middle_Education"),
    ("secondary_share", "Secondary_Education"),
    ("higher_share", "Higher_Education"),
    ("graduate_share", "Graduate_Education"),
    ("other_education_share", "Other_Education"),
    ("illiterate_share", "Illiterate_Education"),
]
# Literate_Education is deliberately absent: it is the *subtotal* of the seven
# attainment levels (verified: sum(levels) / Literate_Education = 1.0000), so
# including it would double count. Levels + Illiterate_Education partition the
# education table exactly (ratio 1.0000).

AGE = [
    ("age_0_29_share", "Age_Group_0_29"),
    ("age_30_49_share", "Age_Group_30_49"),
    ("age_50_share", "Age_Group_50"),
    ("age_not_stated_share", "Age not stated"),
]

INCOME = [
    ("income_below_45k_share", "Power_Parity_Less_than_Rs_45000"),
    ("income_45k_90k_share", "Power_Parity_Rs_45000_90000"),
    ("income_90k_150k_share", "Power_Parity_Rs_90000_150000"),
    ("income_150k_240k_share", "Power_Parity_Rs_150000_240000"),
    ("income_240k_330k_share", "Power_Parity_Rs_240000_330000"),
    ("income_330k_425k_share", "Power_Parity_Rs_330000_425000"),
    ("income_425k_545k_share", "Power_Parity_Rs_425000_545000"),
    ("income_above_545k_share", "Power_Parity_Above_Rs_545000"),
]

HH_SIZE = [
    ("hh_size_1_share", "Household_size_1_person_Households"),
    ("hh_size_2_share", "Household_size_2_persons_Households"),
    ("hh_size_3_share", "Household_size_3_persons_Households"),
    ("hh_size_4_share", "Household_size_4_persons_Households"),
    ("hh_size_5_share", "Household_size_5_persons_Households"),
    ("hh_size_6_8_share", "Household_size_6_8_persons_Households"),
    ("hh_size_9plus_share", "Household_size_9_persons_and_above_Households"),
]

# (slug, D1 column, family) for the D5-normalised household indicators.
HOUSEHOLD_INDICATORS = [
    ("lpg_share", "LPG_or_PNG_Households", "cooking_fuel"),
    ("electricity_share", "Housholds_with_Electric_Lighting", "lighting"),
    ("internet_share", "Households_with_Internet", "connectivity"),
    ("computer_share", "Households_with_Computer", "connectivity"),
    ("bicycle_share", "Households_with_Bicycle", "assets"),
    ("car_share", "Households_with_Car_Jeep_Van", "assets"),
    ("radio_share", "Households_with_Radio_Transistor", "assets"),
    ("scooter_share", "Households_with_Scooter_Motorcycle_Moped", "assets"),
    ("tv_share", "Households_with_Television", "assets"),
    ("telephone_share", "Households_with_Telephone_Mobile_Phone", "connectivity"),
    ("mobile_only_share", "Households_with_Telephone_Mobile_Phone_Mobile_only", "connectivity"),
    ("landline_only_share", "Households_with_Telephone_Mobile_Phone_Landline_only", "connectivity"),
    ("both_phone_share", "Households_with_Telephone_Mobile_Phone_Both", "connectivity"),
    ("multi_asset_share", "Households_with_TV_Computer_Laptop_Telephone_mobile_phone_and_Scooter_Car", "assets"),
    ("bathing_facility_share", "Having_bathing_facility_Total_Households", "sanitation"),
    ("no_bathing_share", "Not_having_bathing_facility_within_the_premises_Total_Households", "sanitation"),
    ("bathing_enclosure_no_roof_share", "Type_of_bathing_facility_Enclosure_without_roof_Households", "sanitation"),
    ("latrine_within_premises_share", "Having_latrine_facility_within_the_premises_Total_Households", "sanitation"),
    ("open_defecation_share", "Not_having_latrine_facility_within_the_premises_Alternative_source_Open_Households", "sanitation"),
    ("pit_latrine_share", "Type_of_latrine_facility_Pit_latrine_Households", "sanitation"),
    ("other_latrine_share", "Type_of_latrine_facility_Other_latrine_Households", "sanitation"),
    ("night_soil_open_drain_share", "Type_of_latrine_facility_Night_soil_disposed_into_open_drain_Households", "sanitation"),
    ("flush_other_system_share", "Type_of_latrine_facility_Flush_pour_flush_latrine_connected_to_other_system_Households", "sanitation"),
    ("separate_kitchen_share", "Households_with_separate_kitchen_Cooking_inside_house", "housing"),
    ("other_cooking_fuel_share", "Type_of_fuel_used_for_cooking_Any_other_Households", "cooking_fuel"),
    ("tapwater_share", "Main_source_of_drinking_water_Tapwater_Households", "water"),
    ("handpump_share", "Main_source_of_drinking_water_Handpump_Tubewell_Borewell_Households", "water"),
    ("tubewell_borehole_share", "Main_source_of_drinking_water_Tubewell_Borehole_Households", "water"),
    ("uncovered_well_share", "Main_source_of_drinking_water_Un_covered_well_Households", "water"),
    ("spring_share", "Main_source_of_drinking_water_Spring_Households", "water"),
    ("river_canal_share", "Main_source_of_drinking_water_River_Canal_Households", "water"),
    ("tank_pond_lake_share", "Main_source_of_drinking_water_Tank_Pond_Lake_Households", "water"),
    ("other_water_source_share", "Main_source_of_drinking_water_Other_sources_Households", "water"),
    ("water_within_premises_share", "Location_of_drinking_water_source_Within_the_premises_Households", "water"),
    ("water_near_premises_share", "Location_of_drinking_water_source_Near_the_premises_Households", "water"),
    ("water_away_share", "Location_of_drinking_water_source_Away_Households", "water"),
]


def _arr(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _share(numerator, denominator) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((den > 0) & np.isfinite(num), num / den, np.nan)


def build_features(d1: pd.DataFrame, d5_hh: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (feature frame, manifest, checks)."""
    out = pd.DataFrame({"key": d1["key"].to_numpy()})
    manifest: list[dict] = []
    checks: list[dict] = []

    def col(name: str) -> np.ndarray:
        if name not in d1.columns:
            raise KeyError(f"missing D1 column: {name}")
        return _arr(d1[name])

    household_denominator = np.asarray(d1["key"].map(d5_hh).to_numpy(dtype=float))

    def add(
        name: str,
        configuration: str,
        family: str,
        values,
        sources: list[str],
        formula: str,
        denominator: str,
        denominator_source: str,
        universe: str,
        notes: str = "",
    ) -> None:
        arr = np.asarray(values, dtype=float)
        if len(arr) != len(out):
            raise ValueError(f"{name}: length {len(arr)} != {len(out)}")
        if name in out.columns:
            raise ValueError(f"duplicate feature name: {name}")
        forbidden = FORBIDDEN_SOURCES.intersection(sources)
        if forbidden:
            raise ValueError(f"{name}: forbidden source columns {forbidden}")
        out[name] = arr
        manifest.append(
            {
                "feature_name": name,
                "configuration": configuration,
                "family": family,
                "source_dataset": "D1 (Census 2011 district abstract)",
                "source_columns": "|".join(sources),
                "formula": formula,
                "denominator": denominator,
                "denominator_source": denominator_source,
                "universe": universe,
                "leakage_status": "clear",
                "notes": notes,
                "retained": True,
                "drop_reason": "",
            }
        )

    def composition(
        block: str,
        configuration: str,
        family: str,
        components: list[tuple[str, str]],
        total: np.ndarray,
        total_label: str,
        total_formula: str,
        universe: str,
        notes: str = "",
    ) -> None:
        """Add a closed composition as shares of its own table total."""
        values = {cname: col(cname) for _, cname in components}
        tot = np.asarray(total, dtype=float)
        summed = np.sum([values[cname] for _, cname in components], axis=0)
        national_ratio = float(np.nansum(summed) / np.nansum(tot))
        additive = abs(national_ratio - 1.0) <= 0.02

        national = {cname: float(np.nansum(values[cname])) for _, cname in components}
        reference = max(national, key=national.get) if additive else None
        checks.append(
            {
                "check": f"composition_additivity:{block}",
                "status": "PASS" if additive else "WARN",
                "details": (
                    f"sum(components)/total = {national_ratio:.4f}; "
                    f"reference category dropped: {reference or 'none (block not additive)'}"
                ),
            }
        )
        for slug, cname in components:
            if cname == reference:
                continue
            add(
                slug,
                configuration,
                family,
                _share(values[cname], tot),
                [cname],
                f"{cname} / {total_label}",
                total_label,
                total_formula,
                universe,
                notes=notes,
            )

    population = col("Population")

    # --- Demography, literacy, caste, work (population denominators) ----------
    add("male_share", CONFIG_A, "demography", _share(col("Male"), population),
        ["Male"], "Male / Population", "Population", "D1 Population", "population")
    add("literacy_rate", CONFIG_A, "education", _share(col("Literate"), population),
        ["Literate"], "Literate / Population", "Population", "D1 Population", "population")
    add("literacy_rate_male", CONFIG_A, "education", _share(col("Male_Literate"), col("Male")),
        ["Male_Literate", "Male"], "Male_Literate / Male", "Male", "D1 Male", "population")
    add("literacy_rate_female", CONFIG_A, "education", _share(col("Female_Literate"), col("Female")),
        ["Female_Literate", "Female"], "Female_Literate / Female", "Female", "D1 Female", "population")
    add("sc_share", CONFIG_A, "caste", _share(col("SC"), population),
        ["SC"], "SC / Population", "Population", "D1 Population", "population")
    add("st_share", CONFIG_A, "caste", _share(col("ST"), population),
        ["ST"], "ST / Population", "Population", "D1 Population", "population")

    add("worker_share", CONFIG_A, "work", _share(col("Workers"), population),
        ["Workers"], "Workers / Population", "Population", "D1 Population", "population")
    add("main_worker_share", CONFIG_A, "work", _share(col("Main_Workers"), population),
        ["Main_Workers"], "Main_Workers / Population", "Population", "D1 Population", "population")
    add("marginal_worker_share", CONFIG_A, "work", _share(col("Marginal_Workers"), population),
        ["Marginal_Workers"], "Marginal_Workers / Population", "Population", "D1 Population", "population")
    add("nonworker_share", CONFIG_A, "work", _share(col("Non_Workers"), population),
        ["Non_Workers"], "Non_Workers / Population", "Population", "D1 Population", "population")

    workers = col("Workers")
    add("cultivator_share_of_workers", CONFIG_A, "work", _share(col("Cultivator_Workers"), workers),
        ["Cultivator_Workers", "Workers"], "Cultivator_Workers / Workers", "Workers", "D1 Workers", "population")
    add("agri_labour_share_of_workers", CONFIG_A, "work", _share(col("Agricultural_Workers"), workers),
        ["Agricultural_Workers", "Workers"], "Agricultural_Workers / Workers", "Workers", "D1 Workers", "population")
    add("household_industry_share_of_workers", CONFIG_A, "work", _share(col("Household_Workers"), workers),
        ["Household_Workers", "Workers"], "Household_Workers / Workers", "Workers", "D1 Workers", "population")
    add("other_worker_share_of_workers", CONFIG_A, "work", _share(col("Other_Workers"), workers),
        ["Other_Workers", "Workers"], "Other_Workers / Workers", "Workers", "D1 Workers", "population")

    # --- Closed compositions -------------------------------------------------
    composition(
        "religion", CONFIG_A, "religion", RELIGION, population, "Population",
        "D1 Population", "population",
    )
    composition(
        "education_attainment", CONFIG_A, "education", EDUCATION, col("Total_Education"),
        "Total_Education", "D1 Total_Education", "population",
        notes=(
            "Mutually exclusive attainment levels plus Illiterate_Education, as shares "
            "of Total_Education. Literate_Education is a subtotal of the levels and is "
            "excluded to avoid double counting."
        ),
    )
    composition("age_structure", CONFIG_A, "demography", AGE, population, "Population",
                "D1 Population", "population")

    income_total = col("Total_Power_Parity")
    composition(
        "income_bands", CONFIG_A, "income", INCOME, income_total, "Total_Power_Parity",
        "D1 Total_Power_Parity", "household",
        notes="Household income bands; shares of the same table's total (no D5 needed).",
    )

    size_components = [cname for _, cname in HH_SIZE]
    size_total = np.sum([col(cname) for cname in size_components], axis=0)
    composition(
        "household_size", CONFIG_A, "household_structure", HH_SIZE, size_total,
        "sum(reported household-size categories)", "sum of D1 household-size categories",
        "household",
        notes="D1 has no household-size total column; the sum of the reported categories is used.",
    )

    owned = col("Ownership_Owned_Households")
    rented = col("Ownership_Rented_Households")
    add("owned_share_of_reported_tenure", CONFIG_A, "tenure", _share(owned, owned + rented),
        ["Ownership_Owned_Households", "Ownership_Rented_Households"],
        "Owned / (Owned + Rented)", "Owned + Rented", "D1 tenure block", "household",
        notes="'Any other' tenure is not reported in D1; share is among reported categories.")

    # --- Household indicators normalised by D5 NO_HH (Configuration B) -------
    for slug, column, family in HOUSEHOLD_INDICATORS:
        add(
            slug,
            CONFIG_B,
            family,
            _share(col(column), household_denominator),
            [column],
            f"{column} / D5 NO_HH",
            "D5 NO_HH (district households, Census 2011 PCA)",
            "D5 census_2011.csv, Rural + Urban summed",
            "household",
            notes="Missing for the 12 districts whose name occurs in two states.",
        )

    manifest_frame = pd.DataFrame(
        [
            {
                **row,
                "n_non_null": int(out[row["feature_name"]].notna().sum()),
                "pct_non_null": round(100 * out[row["feature_name"]].notna().mean(), 2),
            }
            for row in manifest
        ]
    )
    return out, manifest_frame, checks


def screen_features(
    features: pd.DataFrame,
    target: pd.Series,
    manifest: pd.DataFrame,
    checks: list[dict],
    config_a: list[str],
    config_b_only: list[str],
    subset_mask: np.ndarray,
    collinearity_threshold: float = 0.98,
    target_correlation_threshold: float = 0.90,
) -> tuple[list[str], list[str], pd.DataFrame, list[dict]]:
    """Collinearity and target-correlation screens.

    Correlations are computed on the full sample, before any modelling, and used
    only to freeze a feature set. They never look at cross-validation folds, so
    no fold information leaks; the trade-off is documented in the manifest.
    """
    drops: list[tuple[str, str, float, str]] = []
    new_checks: list[dict] = []

    def screen(candidates: list[str], existing: list[str], frame: pd.DataFrame) -> list[str]:
        kept = list(existing)
        for name in candidates:
            column = frame[name]
            if not column.notna().any():
                drops.append((name, "", 0.0, "all missing"))
                continue
            reference = kept + [k for k in candidates[: candidates.index(name)] if k in kept]
            if reference:
                corr = frame[reference].corrwith(column).abs()
                if corr.notna().any() and float(corr.max()) > collinearity_threshold:
                    worst = str(corr.idxmax())
                    drops.append((name, worst, round(float(corr.max()), 4),
                                  f"|r| > {collinearity_threshold} with {worst}"))
                    continue
            kept.append(name)
        return kept

    kept_a = screen(config_a, [], features)
    kept_b = screen(config_b_only, kept_a, features.loc[subset_mask])

    new_checks.append({
        "check": "collinearity_screen",
        "status": "PASS",
        "details": (
            f"threshold |r| > {collinearity_threshold}; "
            f"{len(drops)} feature(s) dropped; A kept {len(kept_a)} of {len(config_a)}"
        ),
    })

    # Target-correlation (indirect leakage) screen.
    correlations = {}
    for name in kept_a + [c for c in kept_b if c not in kept_a]:
        pair = pd.concat([features.loc[subset_mask, name], target.loc[subset_mask]], axis=1).dropna()
        correlations[name] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 2 else np.nan
    worst_name = max(correlations, key=lambda k: abs(correlations[k]) if np.isfinite(correlations[k]) else -1)
    worst_value = correlations[worst_name]
    flags = [k for k, v in correlations.items() if np.isfinite(v) and abs(v) > target_correlation_threshold]

    new_checks.append({
        "check": "target_correlation_screen",
        "status": "PASS" if not flags else "WARN",
        "details": (
            f"threshold |r| > {target_correlation_threshold}; "
            f"max |r| = {abs(worst_value):.3f} ({worst_name}); flagged: {flags or 'none'}"
        ),
    })

    for name, worst, value, reason in drops:
        manifest.loc[manifest["feature_name"] == name, "retained"] = False
        manifest.loc[manifest["feature_name"] == name, "drop_reason"] = reason
        manifest.loc[manifest["feature_name"] == name, "notes"] = (
            manifest.loc[manifest["feature_name"] == name, "notes"].fillna("").astype(str)
            + f" | redundant with {worst} (|r|={value})"
        ).str.strip(" |")

    for name in flags:
        manifest.loc[manifest["feature_name"] == name, "leakage_status"] = "REVIEW"

    screening = {
        "kept_a": kept_a,
        "kept_b_only": [c for c in kept_b if c not in kept_a],
        "drops": [{"feature": n, "collinear_with": w, "abs_r": v, "reason": r} for n, w, v, r in drops],
        "target_correlations": {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in correlations.items()},
        "flagged": flags,
    }
    return kept_a, screening["kept_b_only"], manifest, new_checks
