"""Create a leakage-aware train/test split using valid targets and GameID groups."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from softball_xrv.config import (
    CLEAN_DATA_PATH,
    TRAIN_PATH,
    TEST_PATH,
    TARGET_COL,
    GROUP_COL,
    RANDOM_STATE,
    TEST_SIZE,
)


def main() -> None:

    df = pd.read_csv(CLEAN_DATA_PATH)

    required_cols = [TARGET_COL, GROUP_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in clean data: {missing_cols}")

    rows_before_target = len(df)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    print("Rows dropped for missing/invalid target:", rows_before_target - len(df))

    rows_before_gameid = len(df)
    df[GROUP_COL] = df[GROUP_COL].astype("string").str.strip()
    valid_gameid = df[GROUP_COL].str.fullmatch(r"\d{8}-[A-Za-z0-9]+-\d+", na=False)
    df = df[valid_gameid].copy()
    print("Rows dropped for invalid GameID:", rows_before_gameid - len(df))

    if df.empty:
        raise ValueError("No rows remain after filtering valid target and GameID.")

    if df[GROUP_COL].nunique() < 2:
        raise ValueError("Need at least 2 unique GameID groups for GroupShuffleSplit.")

    # Split by GameID so pitches from the same game cannot appear in both train and test.
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    shared_games = set(train_df[GROUP_COL]).intersection(set(test_df[GROUP_COL]))
    if shared_games:
        raise ValueError(f"GameID leakage detected: {len(shared_games)} shared games")

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train unique GameID:", train_df[GROUP_COL].nunique())
    print("Test unique GameID:", test_df[GROUP_COL].nunique())
    print("Shared GameID between train/test:", len(shared_games))


if __name__ == "__main__":
    main()
