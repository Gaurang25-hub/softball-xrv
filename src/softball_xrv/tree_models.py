"""Build, preprocess, and fit tree-based classifiers for grouped xRV validation."""

import numpy as np
import pandas as pd
from collections.abc import Sequence

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30


TREE_MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
}


def get_tree_models() -> dict[str, object]:
    """Return fresh tree model objects keyed by machine-friendly model names."""

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            criterion="log_loss",
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_median_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """Fit median imputer on training fold only."""

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    return imputer


def transform_with_imputer(
    imputer: SimpleImputer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Apply fitted imputer and preserve column names."""

    return pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )


def preprocess_tree_features(
    train_part_df: pd.DataFrame,
    val_part_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce, clean, fit train-fold imputer, and transform train/validation features."""

    X_train = train_part_df[list(feature_cols)].copy()
    X_val = val_part_df[list(feature_cols)].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    # RandomForest can handle NaNs natively, but we impute deliberately
    # to standardize preprocessing across RF, CatBoost, and XGBoost.
    imputer = fit_median_imputer(X_train)

    X_train = transform_with_imputer(imputer, X_train)
    X_val = transform_with_imputer(imputer, X_val)

    return X_train, X_val


def fit_tree_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> object:
    """Fit one tree model, using early stopping where supported."""

    if model_name not in TREE_MODEL_DISPLAY_NAMES:
        valid_names = ", ".join(TREE_MODEL_DISPLAY_NAMES)
        raise ValueError(
            f"Unknown tree model '{model_name}'. Expected one of: {valid_names}."
        )

    if model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    elif model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

    else:
        model.fit(X_train, y_train)

    return model
