"""Coerce Track A pitch-tracking features and write raw-data quality reports."""

import pandas as pd
from softball_xrv.config  import CLEAN_DATA_PATH, RAW_CSV_PATH, REPORTS_DIR
from softball_xrv.schema import TRACK_A_COLS


def main() -> None:
    quality_dir = REPORTS_DIR / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV_PATH)

    # Coerce only Track A features; retain all raw columns for traceability.
    missing_cols = [col for col in TRACK_A_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing Track A columns in raw data: {missing_cols}")

    for col in TRACK_A_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_ratio = df[TRACK_A_COLS].isna().mean().sort_values(ascending=False)
    missing_ratio.rename("missing_ratio").to_csv(
        quality_dir / "missing_ratio_track_a.csv",
        index_label="feature",
    )

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"Saved cleaned data: {CLEAN_DATA_PATH}")
    print(f"Saved missingness report: {quality_dir / 'missing_ratio_track_a.csv'}")


if __name__ == "__main__":
    main()
