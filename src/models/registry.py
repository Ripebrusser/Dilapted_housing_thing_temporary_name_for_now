"""Model registry and search spaces for Phase 3.

Every estimator is wrapped in a scikit-learn ``Pipeline`` so that imputation and
scaling are fitted **inside** the training folds only. Search spaces are compact
and justified for 640 observations: no grid exceeds 12 candidates, and every
parameter has a documented reason for being in the grid.

``n_configs`` is the number of hyperparameter candidates per outer fold, i.e. the
size of the grid crossed with the inner CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

SEED = 42
N_ESTIMATORS = 200


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict
    scaling: bool
    complexity_rank: int
    notes: str

    @property
    def n_configs(self) -> int:
        """Number of grid points only.

        The inner cross-validation multiplies this by its fold count; the value
        here is the grid size, not the number of fits performed.
        """
        if not self.param_grid:
            return 1
        return prod(len(values) for values in self.param_grid.values())


def _pipeline(model, scaling: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def build_models(seed: int = SEED) -> dict[str, ModelSpec]:
    """Return the ten approved model families in a stable order."""
    return {
        "dummy_mean": ModelSpec(
            name="dummy_mean",
            estimator=_pipeline(DummyRegressor(strategy="mean"), False),
            param_grid={},
            scaling=False,
            complexity_rank=1,
            notes="Predicts the training-fold mean. The floor every model must beat.",
        ),
        "ridge": ModelSpec(
            name="ridge",
            estimator=_pipeline(Ridge(), True),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            scaling=True,
            complexity_rank=2,
            notes="L2-penalised linear model; alpha spans weak to strong shrinkage for 43-77 correlated shares.",
        ),
        "elastic_net": ModelSpec(
            name="elastic_net",
            estimator=_pipeline(ElasticNet(max_iter=50000, random_state=seed), True),
            param_grid={
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
            scaling=True,
            complexity_rank=3,
            notes="L1+L2; handles the compositional blocks by shrinking redundant members to zero.",
        ),
        "knn": ModelSpec(
            name="knn",
            estimator=_pipeline(KNeighborsRegressor(), True),
            param_grid={
                "model__n_neighbors": [5, 10, 15, 25],
                "model__weights": ["uniform", "distance"],
            },
            scaling=True,
            complexity_rank=3,
            notes="Carries over the Phase 2 baseline with tuning; distance-weighted neighbours on scaled shares.",
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            estimator=_pipeline(DecisionTreeRegressor(random_state=seed), False),
            param_grid={
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_leaf": [2, 5, 10],
            },
            scaling=False,
            complexity_rank=4,
            notes="Single tree; depth and leaf size control overfitting on 640 rows.",
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=_pipeline(
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1
                ),
                False,
            ),
            param_grid={
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", 0.5],
            },
            scaling=False,
            complexity_rank=6,
            notes="Bagged trees; max_features controls decorrelation at 43-77 features.",
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            estimator=_pipeline(
                ExtraTreesRegressor(
                    n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1
                ),
                False,
            ),
            param_grid={
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", 0.5],
            },
            scaling=False,
            complexity_rank=6,
            notes="Extremely randomised trees; lower variance than a single tree.",
        ),
        "gradient_boosting": ModelSpec(
            name="gradient_boosting",
            estimator=_pipeline(
                GradientBoostingRegressor(random_state=seed, subsample=0.8), False
            ),
            param_grid={
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
            scaling=False,
            complexity_rank=7,
            notes="Shallow boosted trees with 80% subsampling; depth 2-3 suits a small n.",
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            estimator=_pipeline(
                HistGradientBoostingRegressor(
                    random_state=seed, early_stopping=False
                ),
                False,
            ),
            param_grid={
                "model__max_iter": [150, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [7, 15],
            },
            scaling=False,
            complexity_rank=7,
            notes="Histogram boosting; early_stopping disabled so no internal split consumes training data.",
        ),
        "svr_rbf": ModelSpec(
            name="svr_rbf",
            estimator=_pipeline(SVR(kernel="rbf"), True),
            param_grid={
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.1, 0.5],
                "model__gamma": ["scale"],
            },
            scaling=True,
            complexity_rank=5,
            notes="RBF kernel; C and epsilon tuned, gamma fixed at 'scale' to keep the grid small.",
        ),
    }


def describe_models(models: dict[str, ModelSpec] | None = None) -> list[dict]:
    """Serialisable description of every model and its search space."""
    models = models or build_models()
    return [
        {
            "model": spec.name,
            "n_configs": spec.n_configs,
            "scaling": spec.scaling,
            "param_grid": spec.param_grid,
            "complexity_rank": spec.complexity_rank,
            "notes": spec.notes,
        }
        for spec in models.values()
    ]
