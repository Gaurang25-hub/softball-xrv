"""Run train-only EDA for Track A baseline columns, domain-engineered columns,
and EDA-proposed candidate columns. Save plots and summary artifacts under
reports/eda.
"""

from collections.abc import Sequence
from pathlib import Path
import numpy as np
import pandas as pd
from softball_xrv.schema import TRACK_A_COLS
from softball_xrv.config import TRAIN_PATH, REPORTS_DIR, TARGET_COL
from softball_xrv.plotting import (
    distribution_plots,
    feature_vs_target_scatter,
    feature_vs_target_boxplot,
    upper_triangular_pairs,
    save_feature_pair_scatterplots,
)


def load_train_frame(
    train_path: Path, required_cols: Sequence[str], target_col: str
) -> pd.DataFrame:
    """Load the training split and validate required Track A columns."""

    df = pd.read_csv(train_path)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required Track A columns in training data: {missing_cols}")

    if target_col not in df.columns:
        raise KeyError(f"Missing target column in training data: {target_col}")

    return df


def add_domain_engineered_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add core and kinematic domain-engineered columns and return their names."""
    missing_cols = [c for c in TRACK_A_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required Track A columns in training data: {missing_cols}"
        )

    df = df.copy()

    # core_domain_engineered_features
    spin_axis_rad = np.deg2rad(df["SpinAxis"])
    df["SpinAxis_sin"] = np.sin(spin_axis_rad)
    df["SpinAxis_cos"] = np.cos(spin_axis_rad)

    df["DeltaVertAngle"] = df["VertApprAngle"] - df["VertRelAngle"]
    df["DeltaHorzAngle"] = df["HorzApprAngle"] - df["HorzRelAngle"]

    df["DeltaZ"] = df["PlateLocHeight"] - df["RelHeight"]
    safe_zone_time = df["ZoneTime"].replace(0, np.nan)
    df["DeltaZ_over_ZoneTime"] = (
        df["PlateLocHeight"] - df["RelHeight"]
    ) / safe_zone_time

    df["SpinX"] = df["SpinRate"] * df["SpinAxis_cos"]
    df["SpinY"] = df["SpinRate"] * df["SpinAxis_sin"]

    df["VertRelAngle_x_Extension"] = df["VertRelAngle"] * df["Extension"]

    core_domain_cols = [
        "SpinAxis_sin",
        "SpinAxis_cos",
        "DeltaVertAngle",
        "DeltaHorzAngle",
        "DeltaZ",
        "DeltaZ_over_ZoneTime",
        "SpinX",
        "SpinY",
        "VertRelAngle_x_Extension",
    ]

    # (domain) kinematic_engineered_features
    df["AvgSpeedLossRate"] = (df["RelSpeed"] - df["ZoneSpeed"]) / safe_zone_time
    df["AvgVertAngleChangeRate"] = df["DeltaVertAngle"] / safe_zone_time
    df["SpeedLoss"] = df["RelSpeed"] - df["ZoneSpeed"]
    df["SpeedRetention"] = df["ZoneSpeed"] / df["RelSpeed"]

    kinematic_cols = [
        "AvgSpeedLossRate",
        "AvgVertAngleChangeRate",
        "SpeedLoss",
        "SpeedRetention",
    ]

    return df, core_domain_cols, kinematic_cols


def run_feature_target_eda(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scatter_dir: Path,
    boxplot_dir: Path,
) -> None:
    """Save feature-vs-target scatterplots and boxplots for all analysis columns."""
    for feature in feature_cols:
        feature_vs_target_scatter(df, feature, target_col, scatter_dir)
        feature_vs_target_boxplot(df, feature, target_col, boxplot_dir)


def build_correlation_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Compute Pearson and Spearman pair summaries and save the ranked CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    corr_df = df[feature_cols].copy()
    pearson_matrix = corr_df.corr(method="pearson")
    spearman_matrix = corr_df.corr(method="spearman")

    feature_list = corr_df.columns.tolist()

    pearson_pairs = upper_triangular_pairs(pearson_matrix, feature_list)
    spearman_pairs = upper_triangular_pairs(spearman_matrix, feature_list)

    pearson_summary = pd.DataFrame(
        pearson_pairs, columns=["feature_i", "feature_j", "pearson_r"]
    )
    spearman_summary = pd.DataFrame(
        spearman_pairs, columns=["feature_i", "feature_j", "spearman_r"]
    )
    corr_summary = pearson_summary.merge(
        spearman_summary, on=["feature_i", "feature_j"]
    )
    corr_summary["abs_pearson"] = corr_summary["pearson_r"].abs()
    corr_summary["abs_spearman"] = corr_summary["spearman_r"].abs()
    corr_summary["max_abs_corr"] = corr_summary[["abs_pearson", "abs_spearman"]].max(
        axis=1
    )
    corr_summary = corr_summary.sort_values("max_abs_corr", ascending=False)

    corr_summary.to_csv(output_path, index=False)
    return corr_summary


def run_feature_pair_eda(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    output_dir: Path,
) -> None:
    """Save feature-vs-feature scatterplots for manually curated correlation pairs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = sorted({col for pair in pairs for col in pair})
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing feature-pair columns: {missing_cols}")

    save_feature_pair_scatterplots(df, pairs, output_dir)


def add_high_interest_candidate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add proposed high-interest candidate features."""

    required_cols = [
        "HorzBreak",
        "DeltaHorzAngle",
        "InducedVertBreak",
        "AvgVertAngleChangeRate",
        "ZoneTime",
        "RelSpeed",
        "ZoneSpeed",
        "SpinAxis_cos",
        "SpinAxis_sin",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing prerequisite columns for candidate feature engineering: {missing_cols}"
        )

    df = df.copy()

    df["abs_HorzBreak"] = df["HorzBreak"].abs()
    df["abs_DeltaHorzAngle"] = df["DeltaHorzAngle"].abs()
    df["abs_InducedVertBreak"] = df["InducedVertBreak"].abs()
    df["abs_AvgVertAngleChangeRate"] = df["AvgVertAngleChangeRate"].abs()

    safe_zone_time = df["ZoneTime"].replace(0, np.nan)
    df["ZoneTime_reciprocal"] = 1 / safe_zone_time

    df["RelSpeed_x_ZoneTime"] = df["RelSpeed"] * df["ZoneTime"]
    df["ZoneSpeed_x_ZoneTime"] = df["ZoneSpeed"] * df["ZoneTime"]

    df["SpinAxis_cos_pos"] = df["SpinAxis_cos"].where(df["SpinAxis_cos"] > 0, 0)
    df["SpinAxis_cos_neg"] = df["SpinAxis_cos"].where(df["SpinAxis_cos"] < 0, 0)
    df["SpinAxis_sin_neg"] = df["SpinAxis_sin"].where(df["SpinAxis_sin"] < 0, 0)
    df["SpinAxis_sin_pos"] = df["SpinAxis_sin"].where(df["SpinAxis_sin"] > 0, 0)

    return df


def main() -> None:

    eda_dir = REPORTS_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_frame(TRAIN_PATH, TRACK_A_COLS, TARGET_COL)

    df, core_domain_cols, kinematic_cols = add_domain_engineered_features(df)

    distribution_dir = eda_dir / "distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    inspection_cols = ["PlateLocHeight", "DeltaZ", "DeltaZ_over_ZoneTime"]

    for col in inspection_cols:
        distribution_plots(df, col, distribution_dir)

    scatter_dir = eda_dir / "feature_target" / "scatterplots"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    boxplot_dir = eda_dir / "feature_target" / "boxplots"
    boxplot_dir.mkdir(parents=True, exist_ok=True)

    analysis_cols = TRACK_A_COLS + core_domain_cols + kinematic_cols
    run_feature_target_eda(
        df,
        analysis_cols,
        TARGET_COL,
        scatter_dir,
        boxplot_dir,
    )
    feature_feature_dir = eda_dir / "feature_feature"
    correlation_summary_path = feature_feature_dir / "correlation_summary.csv"

    build_correlation_summary(df, analysis_cols, correlation_summary_path)

    high_interest_pairs = [
        ("ZoneSpeed", "ZoneTime"),
        ("RelSpeed", "ZoneTime"),
        ("SpinAxis_sin", "DeltaHorzAngle"),
        ("HorzBreak", "SpinAxis_sin"),
        ("DeltaHorzAngle", "SpinY"),
        ("HorzBreak", "SpinY"),
        ("InducedVertBreak", "SpinX"),
        ("InducedVertBreak", "SpinAxis_cos"),
        ("InducedVertBreak", "AvgVertAngleChangeRate"),
        ("SpinAxis_cos", "AvgVertAngleChangeRate"),
        ("SpinX", "AvgVertAngleChangeRate"),
    ]
    high_interest_pairs_dir = eda_dir / "feature_feature_scatter" / "high_interest"

    run_feature_pair_eda(df, high_interest_pairs, high_interest_pairs_dir)

    primary_candidates = [
        "abs_HorzBreak",
        "abs_DeltaHorzAngle",
        "abs_InducedVertBreak",
        "abs_AvgVertAngleChangeRate",
        "ZoneTime_reciprocal",
        "RelSpeed_x_ZoneTime",
        "ZoneSpeed_x_ZoneTime",
    ]

    secondary_candidates = [
        "SpinAxis_cos_pos",
        "SpinAxis_cos_neg",
        "SpinAxis_sin_neg",
        "SpinAxis_sin_pos",
    ]

    df = add_high_interest_candidate_features(df)

    primary_candidates_dir = eda_dir / "candidates" / "primary"
    primary_candidates_dir.mkdir(parents=True, exist_ok=True)

    for feature in primary_candidates:
        feature_vs_target_boxplot(df, feature, TARGET_COL, primary_candidates_dir)

    secondary_candidates_dir = eda_dir / "candidates" / "secondary"
    secondary_candidates_dir.mkdir(parents=True, exist_ok=True)

    for feature in secondary_candidates:
        feature_vs_target_boxplot(df, feature, TARGET_COL, secondary_candidates_dir)


if __name__ == "__main__":
    main()
