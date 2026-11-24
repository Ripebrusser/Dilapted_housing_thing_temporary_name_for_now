"""Fit and persist the ratified final model: Ridge Regression, Configuration B.

This is the deployment side of the project. It does **not** re-select anything:
the model, the alpha, the feature schema and the target were ratified in
Phase 3 and are only assembled here.

Ratified configuration
----------------------
* configuration ``B_augmented_628`` — 628 districts with a resolvable household
  denominator, 77 predictors in the frozen manifest order;
* preprocessing ``SimpleImputer(median) -> StandardScaler``, fitted with the
  model (a no-op imputer in practice: Configuration B has no missing values);
* ``Ridge(alpha = 100.0)``;
* target ``target_deprivation_rate_pct`` — the share of dilapidated census
  houses in a district, 2011 Census.

What the fitted model's in-sample predictions are **not**
----------------------------------------------------------
They are not a performance estimate. Fitting on all 628 districts and scoring
those same districts reproduces the training data, so the residuals are
optimistically small by construction. The generalisation estimate remains the
grouped nested out-of-fold evaluation in ``evaluation/metrics/`` — MAE 1.749,
RMSE 2.270, R2 0.399 (fold-mean) / 0.468 (pooled), Spearman 0.631. Every file
this module writes is labelled ``refit_in_sample`` for that reason.

Security note
-------------
``joblib``/``pickle`` payloads execute code on load. Load
``models/final/ridge_b_pipeline.joblib`` only from this repository, and never
from a downloaded or shared archive.

Usage: py -3.14 src/prediction/final_model.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.run_nested_cv import build_configurations  # noqa: E402

MODEL_DIR = ROOT / "models" / "final"
MODEL_PATH = MODEL_DIR / "ridge_b_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "ridge_b_metadata.json"
MANIFEST_PATH = MODEL_DIR / "feature_manifest.csv"
TRAINING_CONFIG_PATH = MODEL_DIR / "training_config.json"

CONFIGURATION = "B_augmented_628"
MODEL_ID = "ridge_b_alpha100_v1"
ALPHA = 100.0
SEED = 42
N_DISTRICTS = 628
N_FEATURES = 77

TARGET_DEFINITION = {
    "column": "target_deprivation_rate_pct",
    "construction": (
        "100 * dilapidated census houses / total census houses, "
        "from D2 (HLPCA) 'Total Number of Dilapidated' on the Rural/Urban == 'Total' row"
    ),
    "units": "percent of census houses in the district",
    "vintage": "2011 Census of India",
    "scope": (
        "district-level prevalence of dilapidated census houses; "
        "NOT a measure of individual homelessness"
    ),
    "not_used": "D1 'Households' (330.8M census houses, not 247M households)",
}

RAW_SOURCES = (
    "datasets/raw/india_census/india-districts-census-2011.csv",
    "datasets/raw/india_census/india_census_housing-hlpca-full.csv",
    "datasets/raw/india_pca/census_2011.csv",
)


# --------------------------------------------------------------------------- #
# data and fingerprints
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approved_training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return the ratified (X, y, keys, groups) for Configuration B.

    Fails loudly rather than silently proceeding on a different schema.
    """
    config = build_configurations(["b628"])["b628"]
    X = config["X"]
    y = np.asarray(config["y"], dtype=float)
    keys = np.asarray(config["keys"])
    groups = np.asarray(config["groups"])

    if X.shape != (N_DISTRICTS, N_FEATURES):
        raise SystemExit(f"STOP: expected {N_DISTRICTS}x{N_FEATURES} features, got {X.shape}")
    if len(y) != N_DISTRICTS or len(keys) != N_DISTRICTS or len(groups) != N_DISTRICTS:
        raise SystemExit("STOP: feature and target lengths disagree with the ratified configuration")
    if X.isna().any().any():
        raise SystemExit("STOP: Configuration B must have no missing feature values")
    if not np.all(np.isfinite(X.to_numpy(dtype=float))):
        raise SystemExit("STOP: non-finite value in the feature matrix")
    if not np.all(np.isfinite(y)):
        raise SystemExit("STOP: non-finite target value")
    if TARGET_DEFINITION["column"] in X.columns:
        raise SystemExit("STOP: the target column is present among the predictors")
    return X, y, keys, groups


def feature_fingerprint(X: pd.DataFrame) -> str:
    """Stable hash of the feature matrix: names, order and values."""
    canonical = X.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def feature_order_hash(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


def dependency_versions() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:  # pragma: no cover - environment problem
            versions[package] = "not installed"
    versions["python"] = sys.version.split()[0]
    return versions


# --------------------------------------------------------------------------- #
# fit / save / load
# --------------------------------------------------------------------------- #


def build_pipeline() -> Pipeline:
    """The exact preprocessing and estimator used by the validated pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ALPHA)),
        ]
    )


def fit_final_model() -> tuple[Pipeline, dict]:
    X, y, keys, groups = approved_training_data()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    features = list(X.columns)
    manifest_sha = hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()
    sources = {}
    for relative in RAW_SOURCES:
        path = ROOT / relative
        sources[relative] = sha256_file(path) if path.exists() else "MISSING"

    metadata = {
        "model_id": MODEL_ID,
        "model_type": "Ridge regression (scikit-learn), alpha = 100.0",
        "model_version": "1.0.0",
        "configuration": CONFIGURATION,
        "alpha": ALPHA,
        "random_state": SEED,
        "n_training_districts": int(len(y)),
        "n_features": int(len(features)),
        "n_states": int(len(np.unique(groups))),
        "features": features,
        "feature_order_sha256": feature_order_hash(features),
        "feature_matrix_sha256": feature_fingerprint(X),
        "feature_manifest_sha256": manifest_sha,
        "training_target_sha256": hashlib.sha256(
            pd.Series(y).to_csv(index=False, header=False, float_format="%.17g").encode()
        ).hexdigest(),
        "raw_source_sha256": sources,
        "target_definition": TARGET_DEFINITION,
        "preprocessing": "SimpleImputer(strategy='median') then StandardScaler, fitted with the model",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dependency_versions": dependency_versions(),
        "ratio_features_note": (
            "five Configuration B rates exceed 1.0 by up to 6.2% (denominator universe "
            "mismatch, documented in Phase 2); they are used as-is and never clipped at "
            "fit or inference time"
        ),
        "range_constraint": (
            "Ridge is unconstrained: a fitted value may fall slightly outside the physically "
            "admissible [0, 100] percentage range. Values are reported exactly as computed and "
            "flagged in the prediction file; they are never clipped."
        ),
        "predictions_are_not_validation": True,
        "validation_reference": (
            "grouped nested CV, 5 outer x 4 inner folds, Phase 3: "
            "MAE 1.7490, RMSE 2.2703, R2 0.3992 (fold-mean) / 0.4680 (pooled), Spearman 0.6672"
        ),
        "persistence_warning": (
            "joblib uses pickle and can execute arbitrary code on load. Only load this file "
            "from this repository; never from an untrusted source."
        ),
    }
    return pipeline, metadata


def save_artifacts(pipeline: Pipeline, metadata: dict) -> dict[str, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    pd.DataFrame(
        {"position": range(1, len(metadata["features"]) + 1), "feature": metadata["features"]}
    ).to_csv(MANIFEST_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    TRAINING_CONFIG_PATH.write_text(
        json.dumps(
            {
                "configuration": CONFIGURATION,
                "alpha": ALPHA,
                "model_id": MODEL_ID,
                "target_definition": TARGET_DEFINITION,
                "preprocessing": metadata["preprocessing"],
                "feature_order_sha256": metadata["feature_order_sha256"],
                "dependency_versions": metadata["dependency_versions"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": MODEL_PATH,
        "metadata": METADATA_PATH,
        "manifest": MANIFEST_PATH,
        "training_config": TRAINING_CONFIG_PATH,
    }


def load_artifacts(model_path: Path = MODEL_PATH) -> tuple[Pipeline, dict]:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"no persisted model at {model_path}; run `py -3.14 src/prediction/final_model.py` first"
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def main() -> int:
    pipeline, metadata = fit_final_model()
    paths = save_artifacts(pipeline, metadata)

    reloaded, metadata_again = load_artifacts()
    X, y, _, _ = approved_training_data()
    before = pipeline.predict(X)
    after = reloaded.predict(X)
    difference = float(np.max(np.abs(before - after)))
    if difference > 1e-10:
        raise SystemExit(f"STOP: reloaded model does not reproduce predictions (max |d| {difference:.3e})")
    if metadata_again["feature_order_sha256"] != metadata["feature_order_sha256"]:
        raise SystemExit("STOP: feature order differs between save and load")

    print("Final model fitted and persisted")
    for label, path in paths.items():
        print(f"  {label:16} {path.relative_to(ROOT)}  ({Path(path).stat().st_size:,} bytes)")
    print(f"  districts        {len(y)}   features {X.shape[1]}   states {metadata['n_states']}")
    print(f"  alpha            {metadata['alpha']}")
    print(f"  reload identical max |d| = {difference:.1e}")
    print("  IN-SAMPLE fit range: "
          f"{before.min():.3f} to {before.max():.3f} pp "
          "(descriptive only; NOT a performance estimate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
