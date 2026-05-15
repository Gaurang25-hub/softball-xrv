"""Run grouped cross-validation model selection on train.csv only."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from softball_xrv.config import TRAIN_PATH, REPORTS_DIR, TARGET_COL, GROUP_COL, RANDOM_STATE
from softball_xrv.feature_sets import FEATURE_SETS, add_modeling_features, validate_modeling_features
from softball_xrv.metrics import (
    exact_match_accuracy,
    balanced_accuracy,
    multiclass_log_loss,
    top_k_accuracy,
    mean_absolute_class_error,
    mean_absolute_xrv_error,
    build_classification_report,
    build_confusion_matrix,
    save_classification_report,
    save_confusion_matrix,
)
from softball_xrv.tree_models import (
    TREE_MODEL_DISPLAY_NAMES,
    EARLY_STOPPING_ROUNDS,
    get_tree_models,
    preprocess_tree_features,
)
from softball_xrv.nn_train import NN_EXPERIMENTS, NNTrainConfig, train_nn_one_fold


N_SPLITS = 5
TOP_K = 2
OUT_DIR = REPORTS_DIR / "model_selection"

NN_MODEL_KEYS = {
    "PyTorch_HardCE_plus_ExpectedDistance": "nn_hard_ce_distance",
    "PyTorch_SoftLabelCE": "nn_soft_label_ce",
    "PyTorch_HybridCE_SharpSoftCE_Distance": "nn_hybrid_ce_sharp_soft_distance",
}


def load_train():
    """Load train.csv, add modeling features, and create ordered class IDs."""

    df = pd.read_csv(TRAIN_PATH)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise")
    df[GROUP_COL] = df[GROUP_COL].astype("string").str.strip()

    df = add_modeling_features(df)
    validate_modeling_features(df)

    xrv_class_values = np.sort(df[TARGET_COL].unique()).astype(float)
    df["target_class"] = pd.Categorical(
        df[TARGET_COL],
        categories=xrv_class_values,
        ordered=True,
    ).codes.astype(np.int64)

    if (df["target_class"] < 0).any():
        raise ValueError("Target encoding failed.")

    class_labels = np.arange(len(xrv_class_values), dtype=np.int64)
    return df, class_labels, xrv_class_values



def score_predictions(y_true, y_pred, y_proba, class_labels, xrv_class_values):
    """Compute the main model-selection metrics for one validation output."""

    k = min(TOP_K, len(class_labels))
    return {
        "exact_accuracy": exact_match_accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "log_loss": multiclass_log_loss(y_true, y_proba, class_labels),
        f"top_{k}_accuracy": top_k_accuracy(y_true, y_proba, k=k, class_labels=class_labels),
        "class_step_mae": mean_absolute_class_error(y_true, y_pred),
        "expected_xrv_mae": mean_absolute_xrv_error(y_true, y_proba, xrv_class_values),
    }


def fit_tree_fold(model_name, train_df, val_df, feature_cols, class_labels):
    """
    Fit one tree model on one fold.
    Tree folds use local contiguous labels so CatBoost/XGBoost remain safe
    when a train fold does not contain every global class.
    """
    X_train, X_val = preprocess_tree_features(
        train_part_df=train_df,
        val_part_df=val_df,
        feature_cols=feature_cols,
    )

    y_train_global = train_df["target_class"].to_numpy(dtype=np.int64)
    y_val_global = val_df["target_class"].to_numpy(dtype=np.int64)

    model = get_tree_models()[model_name]

    local_classes = np.sort(np.unique(y_train_global)).astype(np.int64)
    global_to_local = {global_id: i for i, global_id in enumerate(local_classes)}
    y_train_local = np.array([global_to_local[y] for y in y_train_global], dtype=np.int64)

    if model_name == "random_forest":
        model.fit(X_train, y_train_local)

    else:
        seen_mask = np.isin(y_val_global, local_classes)

        if seen_mask.any():
            X_val_seen = X_val.loc[seen_mask]
            y_val_seen_local = np.array(
                [global_to_local[y] for y in y_val_global[seen_mask]],
                dtype=np.int64,
            )

            if model_name == "xgboost":
                model.fit(
                    X_train,
                    y_train_local,
                    eval_set=[(X_val_seen, y_val_seen_local)],
                    verbose=False,
                )
            elif model_name == "catboost":
                model.fit(
                    X_train,
                    y_train_local,
                    eval_set=(X_val_seen, y_val_seen_local),
                    use_best_model=True,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unexpected tree model: {model_name}")

        else:
            if model_name == "xgboost":
                model.set_params(early_stopping_rounds=None)
            model.fit(X_train, y_train_local, verbose=False)

    y_proba_local = np.asarray(model.predict_proba(X_val), dtype=float)

    y_proba = np.zeros((len(X_val), len(class_labels)), dtype=float)
    y_proba[:, local_classes] = y_proba_local
    y_proba = np.clip(y_proba, 1e-15, 1.0)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_pred = y_proba.argmax(axis=1)
    return y_val_global, y_pred, y_proba



def main() -> None:
    """Run model selection and save the main outputs."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, class_labels, xrv_class_values = load_train()

    cv = list(
        StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(
            df,
            y=df["target_class"].to_numpy(),
            groups=df[GROUP_COL].to_numpy(),
        )
    )

    nn_config = NNTrainConfig()
    fold_rows = []
    oof_store = {}

    # Run every feature set against every tree model and every NN experiment.
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        for fold, (train_idx, val_idx) in enumerate(cv, start=1):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            for model_name, display_name in TREE_MODEL_DISPLAY_NAMES.items():
                y_true, y_pred, y_proba = fit_tree_fold(
                    model_name=model_name,
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=feature_cols,
                    class_labels=class_labels,
                )

                metrics = score_predictions(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    class_labels=class_labels,
                    xrv_class_values=xrv_class_values,
                )

                fold_rows.append(
                    {
                        "model_family": "tree",
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "model_name": model_name,
                        "model_display_name": display_name,
                        "fold": fold,
                        "best_epoch": np.nan,
                        "epochs_trained": np.nan,
                        "best_val_loss": np.nan,
                        **metrics,
                    }
                )

                key = (feature_set_name, model_name)
                part = pd.DataFrame(
                    {
                        "true_class": y_true,
                        "pred_class": y_pred,
                    }
                )
                for class_id in class_labels:
                    part[f"proba_{class_id}"] = y_proba[:, class_id]
                oof_store.setdefault(key, []).append(part)

            for display_name, loss_type in NN_EXPERIMENTS.items():
                model_name = NN_MODEL_KEYS[display_name]

                result = train_nn_one_fold(
                    train_part_df=train_df,
                    val_part_df=val_df,
                    feature_cols=feature_cols,
                    loss_type=loss_type,
                    xrv_class_values=xrv_class_values,
                    target_col="target_class",
                    config=nn_config,
                    verbose=False,
                )

                y_true = np.asarray(result["y_true"], dtype=np.int64)
                y_pred = np.asarray(result["y_pred"], dtype=np.int64)
                y_proba = np.asarray(result["y_proba"], dtype=float)

                metrics = score_predictions(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    class_labels=class_labels,
                    xrv_class_values=xrv_class_values,
                )

                fold_rows.append(
                    {
                        "model_family": "nn",
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "model_name": model_name,
                        "model_display_name": display_name,
                        "fold": fold,
                        "best_epoch": result["best_epoch"],
                        "epochs_trained": result["epochs_trained"],
                        "best_val_loss": result["best_val_loss"],
                        **metrics,
                    }
                )

                key = (feature_set_name, model_name)
                part = pd.DataFrame(
                    {
                        "true_class": y_true,
                        "pred_class": y_pred,
                    }
                )
                for class_id in class_labels:
                    part[f"proba_{class_id}"] = y_proba[:, class_id]
                oof_store.setdefault(key, []).append(part)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "cv_fold_metrics.csv", index=False)

    metric_cols = [
        "exact_accuracy",
        "balanced_accuracy",
        "log_loss",
        f"top_{min(TOP_K, len(class_labels))}_accuracy",
        "class_step_mae",
        "expected_xrv_mae",
        "best_epoch",
        "epochs_trained",
        "best_val_loss",
    ]

    summary_rows = []
    for (feature_set_name, model_name), parts in oof_store.items():
        pair_df = fold_df[
            (fold_df["feature_set"] == feature_set_name)
            & (fold_df["model_name"] == model_name)
        ].copy()

        oof_df = pd.concat(parts, ignore_index=True)
        proba_cols = [f"proba_{class_id}" for class_id in class_labels]

        pooled = score_predictions(
            y_true=oof_df["true_class"].to_numpy(dtype=np.int64),
            y_pred=oof_df["pred_class"].to_numpy(dtype=np.int64),
            y_proba=oof_df[proba_cols].to_numpy(dtype=float),
            class_labels=class_labels,
            xrv_class_values=xrv_class_values,
        )

        row = {
            "model_family": pair_df["model_family"].iloc[0],
            "feature_set": feature_set_name,
            "feature_count": int(pair_df["feature_count"].iloc[0]),
            "model_name": model_name,
            "model_display_name": pair_df["model_display_name"].iloc[0],
        }

        for col in metric_cols:
            row[f"mean_{col}"] = float(pair_df[col].mean())
            row[f"std_{col}"] = float(pair_df[col].std(ddof=0))

        row["oof_exact_accuracy"] = pooled["exact_accuracy"]
        row["oof_balanced_accuracy"] = pooled["balanced_accuracy"]
        row["oof_log_loss"] = pooled["log_loss"]
        row[f"oof_top_{min(TOP_K, len(class_labels))}_accuracy"] = pooled[
            f"top_{min(TOP_K, len(class_labels))}_accuracy"
        ]
        row["oof_class_step_mae"] = pooled["class_step_mae"]
        row["oof_expected_xrv_mae"] = pooled["expected_xrv_mae"]

        summary_rows.append(row)

    leaderboard = pd.DataFrame(summary_rows).sort_values(
        by=[
            "oof_log_loss",
            "oof_expected_xrv_mae",
            "oof_class_step_mae",
            "oof_balanced_accuracy",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard.to_csv(OUT_DIR / "leaderboard.csv", index=False)

    best_per_model = (
        leaderboard.sort_values(
            by=[
                "oof_log_loss",
                "oof_expected_xrv_mae",
                "oof_class_step_mae",
                "oof_balanced_accuracy",
            ],
            ascending=[True, True, True, False],
        )
        .groupby("model_display_name", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_per_model.to_csv(OUT_DIR / "best_per_model.csv", index=False)

    best_row = leaderboard.iloc[0]
    best_key = (best_row["feature_set"], best_row["model_name"])
    best_oof = pd.concat(oof_store[best_key], ignore_index=True)

    best_report = build_classification_report(
        y_true=best_oof["true_class"].to_numpy(dtype=np.int64),
        y_pred=best_oof["pred_class"].to_numpy(dtype=np.int64),
        xrv_class_values=xrv_class_values,
    )
    best_cm = build_confusion_matrix(
        y_true=best_oof["true_class"].to_numpy(dtype=np.int64),
        y_pred=best_oof["pred_class"].to_numpy(dtype=np.int64),
        xrv_class_values=xrv_class_values,
    )

    save_classification_report(best_report, OUT_DIR / "best_classification_report.csv")
    save_confusion_matrix(best_cm, OUT_DIR / "best_confusion_matrix.csv")
    best_row.to_frame().T.to_csv(OUT_DIR / "best_model_selection.csv", index=False)

    print(
        leaderboard[
            [
                "rank",
                "model_display_name",
                "feature_set",
                "oof_log_loss",
                "oof_expected_xrv_mae",
                "oof_balanced_accuracy",
            ]
        ].head(10).to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
