import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from softball_xrv.config import (
    TRAIN_PATH,
    TEST_PATH,
    REPORTS_DIR,
    TARGET_COL,
    GROUP_COL,
    RANDOM_STATE,
    TEST_SIZE,
)

from softball_xrv.feature_sets import (
    FEATURE_SETS,
    add_modeling_features,
    validate_modeling_features,
)

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

from softball_xrv.nn_train import NN_EXPERIMENTS, NNTrainConfig, train_nn_one_fold

TOP_K = 2
OUT_DIR = REPORTS_DIR / "evaluation"

NN_MODEL_KEYS = {
    "PyTorch_HardCE_plus_ExpectedDistance": "nn_hard_ce_distance",
    "PyTorch_SoftLabelCE": "nn_soft_label_ce",
    "PyTorch_HybridCE_SharpSoftCE_Distance": "nn_hybrid_ce_sharp_soft_distance",
}

START_TIME = time.time()


def log(message: str) -> None:
    elapsed_time = time.time() - START_TIME
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} | +{elapsed_time:8.1f}s] {message}", flush=True)


def load_train():
    """Load train.csv, add modeling features, and create ordered class IDs."""

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    processed = []
    for df in (df_train, df_test):
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise")
        df = add_modeling_features(df)
        validate_modeling_features(df)
        processed.append(df)
    df_train, df_test = processed

    xrv_class_values = np.sort(df_train[TARGET_COL].unique()).astype(float)

    for df in (df_train, df_test):
        df["target_class"] = pd.Categorical(
            df[TARGET_COL],
            categories=xrv_class_values,
            ordered=True,
        ).codes.astype(np.int64)

        if (df["target_class"] < 0).any():
            raise ValueError("Target encoding failed.")

    class_labels = np.arange(len(xrv_class_values), dtype=np.int64)
    return df_train, df_test, class_labels, xrv_class_values


def score_predictions(y_true, y_pred, y_proba, class_labels, xrv_class_values):
    """Compute the main model-selection metrics for one validation output."""

    k = min(TOP_K, len(class_labels))
    return {
        "exact_accuracy": exact_match_accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "log_loss": multiclass_log_loss(y_true, y_proba, class_labels),
        f"top_{k}_accuracy": top_k_accuracy(
            y_true, y_proba, k=k, class_labels=class_labels
        ),
        "class_step_mae": mean_absolute_class_error(y_true, y_pred),
        "expected_xrv_mae": mean_absolute_xrv_error(y_true, y_proba, xrv_class_values),
    }


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        "Grouped cross-validation model selection for softball xRV."
    )

    parser.add_argument(
        "--nn-experiments",
        nargs="+",
        choices=["PyTorch_HardCE_plus_ExpectedDistance"],
        default=["PyTorch_HardCE_plus_ExpectedDistance"],
        help="Which NN experiments to run, by display name (default: all).",
    )

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["Set E"],
        default=["Set E"],
        help="Only one set to work with that is Set E",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_feature_sets = {name: FEATURE_SETS[name] for name in args.feature_sets}
    selected_nn_experiments = args.nn_experiments

    log(f"Output dir        : {out_dir}")
    log(f"Features sets     : {list(selected_feature_sets)}")
    log(f"NN experiments    :  {selected_nn_experiments}")

    df_train, df_test, class_labels, xrv_class_values = load_train()

    log(f"Loaded train.csv  : rows={len(df_train)} , classes = {len(class_labels)}")

    nn_config = NNTrainConfig()

    for feature_set_name, feature_cols in selected_feature_sets.items():
        for display_name in selected_nn_experiments:
            loss_type = NN_EXPERIMENTS[display_name]
            model_name = NN_MODEL_KEYS[display_name]
            t0 = time.time()
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
            train_idx, test_idx = next(gss.split(df_train, groups=df_train[GROUP_COL]))

            inner_train_df = df_train.iloc[train_idx].copy()
            inner_val_df = df_train.iloc[test_idx].copy()
            result = train_nn_one_fold(
                train_part_df=inner_train_df,
                val_part_df=inner_val_df,
                test_part_df=df_test,
                feature_cols=feature_cols,
                loss_type=loss_type,
                xrv_class_values=xrv_class_values,
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

            log(
                f"nn='{model_name}' "
                f"set='{feature_set_name}' "
                f"done in {time.time() - t0:6.1f}s "
                f"log_loss={metrics['log_loss']:.4f}"
            )

            pred_df = pd.DataFrame({"true_class": y_true, "pred_class": y_pred})
            for c in range(y_proba.shape[1]):
                pred_df[f"proba_{c}"] = y_proba[:, c]
            pred_df.to_csv(out_dir / "predictions.csv", index=False)

            pd.DataFrame(
                [
                    {
                        **metrics,
                        "best_val_loss": result["best_val_loss"],
                        "best_epoch": result["best_epoch"],
                        "epochs_trained": result["epochs_trained"],
                    }
                ]
            ).to_csv(out_dir / "metrics.csv", index=False)

            save_classification_report(
                build_classification_report(y_true, y_pred, xrv_class_values),
                out_dir / "classification_report.csv",
            )

            save_confusion_matrix(
                build_confusion_matrix(y_true, y_pred, xrv_class_values),
                out_dir / "confusion_matrix.csv",
            )


if __name__ == "__main__":
    main()
