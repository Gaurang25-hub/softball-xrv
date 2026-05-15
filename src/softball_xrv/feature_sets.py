"""Define model-ready feature engineering and curated feature-set variants for xRV experiments."""

from softball_xrv.schema import TRACK_A_COLS
import pandas as pd
import numpy as np


def add_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with deterministic xRV modeling features added."""

    required_cols = [
        "SpinAxis",
        "VertApprAngle",
        "VertRelAngle",
        "HorzApprAngle",
        "HorzRelAngle",
        "PlateLocHeight",
        "RelHeight",
        "ZoneTime",
        "SpinRate",
        "Extension",
        "RelSpeed",
        "ZoneSpeed",
        "InducedVertBreak",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing raw columns required for modeling feature engineering: {missing_cols}"
        )

    df = df.copy()

    spin_axis_rad = np.deg2rad(df["SpinAxis"])

    df["SpinAxis_sin"] = np.sin(spin_axis_rad)
    df["SpinAxis_cos"] = np.cos(spin_axis_rad)

    df["DeltaVertAngle"] = df["VertApprAngle"] - df["VertRelAngle"]
    df["DeltaHorzAngle"] = df["HorzApprAngle"] - df["HorzRelAngle"]

    df["DeltaZ"] = df["PlateLocHeight"] - df["RelHeight"]

    # zero denominators become NaN to avoid invalid division
    safe_zone_time = df["ZoneTime"].replace(0, np.nan)
    df["DeltaZ_over_ZoneTime"] = df["DeltaZ"] / safe_zone_time

    df["SpinX"] = df["SpinRate"] * df["SpinAxis_cos"]
    df["SpinY"] = df["SpinRate"] * df["SpinAxis_sin"]

    df["VertRelAngle_x_Extension"] = df["VertRelAngle"] * df["Extension"]

    df["AvgSpeedLossRate"] = (df["RelSpeed"] - df["ZoneSpeed"]) / safe_zone_time
    df["AvgVertAngleChangeRate"] = df["DeltaVertAngle"] / safe_zone_time

    df["SpeedLoss"] = df["RelSpeed"] - df["ZoneSpeed"]
    safe_rel_speed = df["RelSpeed"].replace(0, np.nan)
    df["SpeedRetention"] = df["ZoneSpeed"] / safe_rel_speed

    df["abs_InducedVertBreak"] = df["InducedVertBreak"].abs()

    return df


def validate_modeling_features(df: pd.DataFrame) -> None:
    """Raise an error if any declared modeling feature is missing from df."""

    missing_cols = [col for col in ALL_MODELING_FEATURES if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing modeling feature columns: {missing_cols}")


BASE_FEATURES = list(TRACK_A_COLS)

CORE_DOMAIN_FEATURES = [
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

KINEMATIC_FEATURES = [
    "AvgSpeedLossRate",
    "AvgVertAngleChangeRate",
    "SpeedLoss",
    "SpeedRetention",
]

DOMAIN_ENGINEERED_FEATURES = CORE_DOMAIN_FEATURES + KINEMATIC_FEATURES

EDA_SURVIVOR_FEATURES = ["abs_InducedVertBreak"]

MODERATE_STANDALONE_SEPARATOR_FEATURES = [
    "DeltaZ_over_ZoneTime",
    "DeltaZ",
    "ZoneSpeed",
    "VertRelAngle",
    "VertApprAngle",
    "DeltaVertAngle",
    "VertRelAngle_x_Extension",
    "SpinX",
    "SpinAxis_cos",
    "RelSpeed",
    "PlateLocHeight",
    "abs_InducedVertBreak",
]

FEATURE_SETS = {
    "Set A": BASE_FEATURES,
    "Set B": BASE_FEATURES + DOMAIN_ENGINEERED_FEATURES,
    "Set C": BASE_FEATURES + DOMAIN_ENGINEERED_FEATURES + EDA_SURVIVOR_FEATURES,
    "Set D": MODERATE_STANDALONE_SEPARATOR_FEATURES,
    # Set E exists to test the EDA survivor as a clean baseline ablation.
    "Set E": BASE_FEATURES + EDA_SURVIVOR_FEATURES,
}

ALL_MODELING_FEATURES = sorted(
    {col for feature_cols in FEATURE_SETS.values() for col in feature_cols}
)
