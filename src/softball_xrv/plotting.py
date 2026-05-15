"""Reusable plotting helpers for train-only EDA outputs."""

from collections.abc import Sequence
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def distribution_plots(
    df: pd.DataFrame,
    feature: str,
    output_dir: Path,
    bins: int = 40,
) -> None:
    """Save histogram, KDE, boxplot, and percentile summary for one feature."""
    x = df[feature].dropna()
    if x.empty:
        print(f"Skipped {feature}: no non-missing values")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(x, bins=bins, kde=True)
    plt.title(f"{feature} distribution train")
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_hist_kde.png", dpi=200)
    plt.close()
    plt.figure(figsize=(8, 2.5))
    sns.boxplot(x=x)
    plt.title(f"{feature} boxplot (train)")
    plt.xlabel(feature)
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_boxplot.png", dpi=200)
    plt.close()
    summary = x.describe(percentiles=[0.25, 0.50, 0.75])
    summary.to_csv(output_dir / f"{feature}_summary.csv", header=["value"])


def feature_vs_target_scatter(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    output_dir: Path,
    alpha: float = 0.35,
    marker_size: int = 12,
) -> None:
    """Save a scatterplot for one feature against the target."""
    pair_df = df[[feature, target_col]].dropna()
    if pair_df.empty:
        print(f"Skipped {feature}: no non-missing feature-target pairs")
        return
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=pair_df, x=feature, y=target_col, alpha=alpha, s=marker_size)
    plt.title(f"{feature} vs {target_col} scatter plot")
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_vs_target.png", dpi=200)
    plt.close()


def feature_vs_target_boxplot(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    output_dir: Path,
) -> None:
    """Save target-level boxplot and summary statistics for one feature."""
    pair_df = df[[target_col, feature]].dropna()
    if pair_df.empty:
        print(f"Skipped {feature}: no non-missing feature-target pairs")
        return
    summary = (
        pair_df.groupby(target_col)[feature]
        .describe(percentiles=[0.25, 0.50, 0.75])
        .sort_index()
    )
    summary[["count", "mean", "25%", "50%", "75%", "min", "max"]].to_csv(
        output_dir / f"{feature}_by_target_summary.csv"
    )
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=pair_df,
        x=target_col,
        y=feature,
        order=sorted(pair_df[target_col].unique()),
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{feature}_by_target_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def upper_triangular_pairs(
    corr_df: pd.DataFrame,
    feature_list: Sequence[str],
) -> list[tuple[str, str, float]]:
    """Return strict upper-triangle feature pairs from a correlation matrix."""
    pairs = []
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            pairs.append((feature_list[i], feature_list[j], float(corr_df.iloc[i, j])))
    return pairs


def save_feature_pair_scatterplots(
    df: pd.DataFrame,
    feature_pairs: Sequence[tuple[str, str]],
    output_dir: Path,
    alpha: float = 0.35,
    marker_size: int = 12,
) -> None:
    """Save scatterplots for selected feature-feature pairs."""
    for x_col, y_col in feature_pairs:
        pair_data = df[[x_col, y_col]].dropna()
        if pair_data.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=pair_data, x=x_col, y=y_col, alpha=alpha, s=marker_size)
        plt.title(f"{x_col} vs {y_col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{x_col}_vs_{y_col}.png", dpi=200)
        plt.close()
