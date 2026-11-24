"""Cross-validation splits for Phase 2 baselines and Phase 3 models.

Primary scheme: K-fold **grouped by state**, so a held-out fold contains states
the model never saw. Reference scheme: shuffled K-fold, repeated with distinct
seeds. The same fold objects are reused for every configuration being compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42


def grouped_folds(groups, n_splits: int = DEFAULT_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds grouped by ``groups`` (state). Deterministic."""
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"cannot build {n_splits} grouped folds from {len(np.unique(groups))} groups"
        )
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (tr, te)
        for tr, te in splitter.split(np.zeros(len(groups)), groups=groups)
    ]


def random_folds(
    n_rows: int,
    n_splits: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Repeated shuffled K-fold, one entry per repeat."""
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for repeat in range(repeats):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        out[f"random_rep{repeat + 1}_seed{seed + repeat}"] = [
            (tr, te) for tr, te in splitter.split(np.arange(n_rows))
        ]
    return out


def assert_folds_valid(
    folds: list[tuple[np.ndarray, np.ndarray]], n_rows: int, groups=None
) -> None:
    """Every row is tested exactly once; train/test never overlap; groups stay whole."""
    test_counts = np.zeros(n_rows, dtype=int)
    for train_idx, test_idx in folds:
        if len(np.intersect1d(train_idx, test_idx)):
            raise AssertionError("fold has overlapping train/test rows")
        test_counts[test_idx] += 1
    if not np.all(test_counts == 1):
        raise AssertionError("test folds do not cover each row exactly once")
    if groups is not None:
        groups = np.asarray(groups)
        for _, test_idx in folds:
            train_groups = set(groups[np.setdiff1d(np.arange(n_rows), test_idx)])
            if set(groups[test_idx]) & train_groups:
                raise AssertionError("a state appears in both train and test")


def fold_table(keys: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]], scheme: str) -> pd.DataFrame:
    rows = []
    for fold_id, (_, test_idx) in enumerate(folds, start=1):
        for idx in test_idx:
            rows.append({"key": keys.iloc[idx], "scheme": scheme, "fold": fold_id})
    return pd.DataFrame(rows)
