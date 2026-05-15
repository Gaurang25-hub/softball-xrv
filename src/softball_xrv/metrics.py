"""Compute probability-aware and ordinal-aware metrics for xRV model selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss,
    top_k_accuracy_score,
    balanced_accuracy_score,
)


def exact_match_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    return float(balanced_accuracy_score(y_true, y_pred))


def multiclass_log_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: np.ndarray,
) -> float:
    """Return multiclass log loss from true class IDs and predicted probabilities."""

    return float(log_loss(y_true, y_proba, labels=class_labels))


def top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int,
    class_labels: np.ndarray,
) -> float:
    """Return top-k accuracy using a fixed full class order."""

    return float(top_k_accuracy_score(y_true, y_proba, k=k, labels=class_labels))


def expected_xrv_from_proba(
    y_proba: np.ndarray,
    xrv_class_values: np.ndarray,
) -> np.ndarray:
    """Return each sample’s probability-weighted expected xRV."""

    return y_proba @ xrv_class_values


def mean_absolute_xrv_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    xrv_class_values: np.ndarray,
) -> float:
    """Measure average absolute miss severity in original xRV units."""

    true_xrv = xrv_class_values[y_true]
    pred_xrv = expected_xrv_from_proba(y_proba, xrv_class_values)

    return float(np.mean(np.abs(pred_xrv - true_xrv)))


def mean_absolute_class_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Measure average ordinal bucket distance between predicted and true classes."""

    return float(np.mean(np.abs(y_pred - y_true)))


def make_class_names(xrv_class_values: np.ndarray) -> list[str]:
    return [
        f"class_{class_id}_xrv_{xrv_value:.4f}"
        for class_id, xrv_value in enumerate(xrv_class_values)
    ]


def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xrv_class_values: np.ndarray,
) -> pd.DataFrame:

    # force full class order so missing validation-fold classes still appear in reports/matrices.
    class_labels = np.arange(len(xrv_class_values))
    class_names = make_class_names(xrv_class_values)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xrv_class_values: np.ndarray,
) -> pd.DataFrame:

    labels = np.arange(len(xrv_class_values))
    class_names = make_class_names(xrv_class_values)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    return pd.DataFrame(cm, index=class_names, columns=class_names)


def save_classification_report(report_df: pd.DataFrame, path: str) -> None:
    report_df.to_csv(path, index=False)


def save_confusion_matrix(cm_df: pd.DataFrame, path: str) -> None:
    cm_df.to_csv(path, index=True)
