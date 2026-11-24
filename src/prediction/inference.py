"""Reusable inference for the ratified Ridge B model.

The point of this module is that scoring a new district table cannot silently
diverge from training: the feature names and their order come from the saved
manifest, and the scaler is the one fitted during training — it is never refit
here.

Schema policy
-------------
* every manifest feature must be present;
* every feature must be numeric;
* missing values are rejected (the training matrix had none, so imputed values
  at inference could not be distinguished from data-entry errors);
* infinite values are rejected;
* extra columns are rejected unless ``allow_extra_columns=True``, in which case
  they are ignored and reported back in ``ignored_columns``;
* output preserves district identifiers so a prediction can always be traced to
  a district.

Identifier columns ``key``, ``district`` and ``state`` are echoed when present.
The district key format is ``"STATE||DISTRICT"``.

Example
-------
>>> from src.prediction.inference import load_scorer, predict_districts
>>> scorer = load_scorer()
>>> frame = <628 x 77 DataFrame with the manifest columns>
>>> predict_districts(frame).head()

Security: the model file is a joblib/pickle payload and can execute code on
load. Load only the file shipped in this repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prediction.final_model import (  # noqa: E402
    METADATA_PATH,
    MODEL_PATH,
    load_artifacts,
)

PREDICTION_COLUMN = "predicted_deprivation_rate_pct"
IDENTIFIER_COLUMNS = ("key", "district", "state")


class SchemaError(ValueError):
    """Raised when an input frame cannot be scored as-is."""


def load_scorer(model_path: Path = MODEL_PATH) -> dict:
    """Load the persisted pipeline and its metadata once, for reuse."""
    pipeline, metadata = load_artifacts(model_path)
    features = list(metadata["features"])
    if len(features) != metadata["n_features"]:
        raise SchemaError("metadata feature list disagrees with n_features")
    # sanity: the fitted scaler must have seen exactly the training rows
    scaler = pipeline.named_steps["scaler"]
    if getattr(scaler, "n_samples_seen_", metadata["n_training_districts"]) != metadata["n_training_districts"]:
        raise SchemaError("the persisted scaler was not fitted on the training sample")
    return {"pipeline": pipeline, "metadata": metadata, "features": features}


def _validate(frame: pd.DataFrame, features: list[str], allow_extra_columns: bool) -> list[str]:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    if frame.empty:
        raise SchemaError("input frame is empty")

    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise SchemaError(
            f"{len(missing)} required feature(s) missing, e.g. {missing[:5]}; "
            f"expected the {len(features)} columns in the saved manifest"
        )

    extra = [column for column in frame.columns if column not in features and column not in IDENTIFIER_COLUMNS]
    if extra and not allow_extra_columns:
        raise SchemaError(
            f"{len(extra)} unexpected column(s): {extra[:5]}. "
            "Pass allow_extra_columns=True to ignore them deliberately."
        )

    block = frame[features]
    non_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(block[column])]
    if non_numeric:
        raise SchemaError(f"non-numeric feature column(s): {non_numeric[:5]}")

    if block.isna().any().any():
        offenders = block.columns[block.isna().any()].tolist()
        raise SchemaError(f"missing values in feature column(s): {offenders[:5]}")

    values = block.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        offenders = block.columns[~np.isfinite(values).all(axis=0)].tolist()
        raise SchemaError(f"infinite values in feature column(s): {offenders[:5]}")

    return extra


def predict_districts(
    frame: pd.DataFrame,
    scorer: dict | None = None,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """Score a district feature table and return identifiers plus predictions.

    Returns a DataFrame with any of ``key``/``district``/``state`` that were
    supplied, the prediction in percentage points, and ``ignored_columns`` when
    extras were tolerated. The input frame is never modified.
    """
    scorer = scorer or load_scorer()
    features = scorer["features"]
    extra = _validate(frame, features, allow_extra_columns)

    ordered = frame.loc[:, features]  # manifest order, always
    predictions = scorer["pipeline"].predict(ordered)
    predictions = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise SchemaError("the model produced a non-finite prediction")

    result = pd.DataFrame(index=frame.index)
    for column in IDENTIFIER_COLUMNS:
        if column in frame.columns:
            result[column] = frame[column]
    result[PREDICTION_COLUMN] = predictions
    result["model_id"] = scorer["metadata"]["model_id"]
    result["prediction_type"] = "final_refit_in_sample(full_628)"
    result["units"] = "percent of dilapidated census houses"
    if extra:
        result["ignored_columns"] = ",".join(extra)
    return result


def main() -> int:
    """Score the training districts as a smoke test of the persisted model."""
    from src.prediction.final_model import approved_training_data

    X, _, keys, groups = approved_training_data()
    frame = X.copy()
    frame.insert(0, "key", keys)
    frame.insert(1, "state", groups)
    frame.insert(2, "district", [key.split("||", 1)[1] for key in keys])

    result = predict_districts(frame)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"model        {metadata['model_id']} (alpha={metadata['alpha']})")
    print(f"scored       {len(result)} districts, {len(metadata['features'])} features")
    print(f"range        {result[PREDICTION_COLUMN].min():.3f} - "
          f"{result[PREDICTION_COLUMN].max():.3f} pp")
    print(f"mean         {result[PREDICTION_COLUMN].mean():.3f} pp")
    print("NOTE: these are in-sample fitted values, not a performance estimate.")
    print(result.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
