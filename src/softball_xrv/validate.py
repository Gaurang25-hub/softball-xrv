"""Validate Track A feature quality against schema bounds for train/test splits."""

from contextlib import redirect_stdout
import pandas as pd
from softball_xrv.schema import TRACK_A_SCHEMA
from softball_xrv.config  import TRAIN_PATH, TEST_PATH, REPORTS_DIR


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty; cannot run validation.")

    print()
    print(f"Validation Report: {dataset_name}")
    for feature, rules in TRACK_A_SCHEMA.items():
        if feature not in df.columns:
            raise KeyError(
                f"{dataset_name} is missing required Track A feature: {feature}"
            )
        print()
        print(f"  {feature} :")
        print()

        print(f"    Description: {rules['description']}")
        print()

        print(f"    Dtype: {rules['dtype']}")
        print()

        print(f"    Unit: {rules['unit']}")
        print()

        col = df[feature]
        missing_mask = col.isna()
        total_count = len(df)
        count_missing = missing_mask.sum()
        decimal_missing = count_missing / total_count
        percent_missing = decimal_missing * 100
        print(f"    Missing rate: {percent_missing:.2f}% ({count_missing} pitches)")
        print()

        missing_test = rules["missingness_rule"]
        if missing_test["max_missing_ratio"] < decimal_missing:
            print(f"     {feature} fails the missingness threshold")

        else:
            print(f"    {feature} passes the missingness threshold")

        print()

        observed = col[~missing_mask]

        # Soft bounds are Tukey-IQR empirical ranges; hard bounds are domain/physics sanity limits.
        # Together they separate typical, suspicious, and strong red-flag pitch-level values.
        hard_bound = rules["hard_bounds"]
        hard_lb = hard_bound["lower"]
        hard_ub = hard_bound["upper"]

        inside_soft_count = 0
        middle_region_count = 0

        if "soft_bounds" in rules:
            soft_bounds = rules["soft_bounds"]
            soft_lb = soft_bounds["lower"]
            soft_ub = soft_bounds["upper"]
            inside_soft = (observed >= soft_lb) & (observed <= soft_ub)
            inside_hard = (observed >= hard_lb) & (observed <= hard_ub)
            middle_region = inside_hard & (~inside_soft)
            inside_soft_count = inside_soft.sum()
            middle_region_count = middle_region.sum()
            inside_soft_percentage = (inside_soft_count / total_count) * 100
            middle_region_percentage = (middle_region_count / total_count) * 100
            print(
                f"    Inside soft interval: {inside_soft_percentage:.2f}% ({inside_soft_count} pitches)"
            )
            print()
            print(
                f"    Inside hard but outside soft: {middle_region_percentage:.2f}% ({middle_region_count} pitches)"
            )
            print()
        else:
            inside_hard = (observed >= hard_lb) & (observed <= hard_ub)
            middle_region = inside_hard
            middle_region_count = middle_region.sum()

        outside_hard = (observed < hard_lb) | (observed > hard_ub)

        outside_hard_count = outside_hard.sum()

        outside_hard_percentage = (outside_hard_count / total_count) * 100
        print(
            f"    Outside hard interval: {outside_hard_percentage:.2f}% ({outside_hard_count} pitches)"
        )
        print()
        if "special_handling" in rules:
            print(f"    Special handling: {rules['special_handling']}")
        print()
        print(
            f"     Category coverage check (missing + soft + middle + outside hard): {count_missing + inside_soft_count + middle_region_count + outside_hard_count} / {total_count} ({(count_missing + inside_soft_count + middle_region_count + outside_hard_count) / total_count:.4f})"
        )
        print()


def main() -> None:

    validation_dir = REPORTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    validation_train_path = validation_dir / "train_validation_report.txt"
    validation_test_path = validation_dir / "test_validation_report.txt"

    datasets = [
        (train_df, "train", validation_train_path),
        (test_df, "test", validation_test_path),
    ]
    # Write each print-based validation report to its corresponding text file.
    for df, dataset_name, report_path in datasets:
        with open(report_path, "w", encoding="utf-8") as file:
            with redirect_stdout(file):
                validate_dataset(df, dataset_name)

    print(f"Saved train validation report: {validation_train_path}")
    print(f"Saved test validation report: {validation_test_path}")


if __name__ == "__main__":
    main()
