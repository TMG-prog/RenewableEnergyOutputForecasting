"""
Creating a shared chronological train-test split for all regression models.

Outputs:
    Project/data/train/X_train.csv
    Project/data/train/X_test.csv
    Project/data/train/y_train.csv
    Project/data/train/y_test.csv
    Project/data/train/split_metadata.json
"""

from pathlib import Path
import json

import pandas as pd


# Configuration:

TRAIN_RATIO = 0.80

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "Project"
    / "data"
    / "cleaned"
    / "processed_features.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "Project"
    / "data"
    / "train"
)

TARGET_COLUMNS = [
    "wind_power_proxy",
    "solar_power_proxy",
]

# These columns directly reveal or reconstruct the targets
# and must be removed to prevent target leakage.
LEAKAGE_COLUMNS = [
    "wind_kph",
    "uv_index",
    "cloud_cover",
    "country_daily_mean_wind",
    "wind_kph_roll_mean_3d",
    "wind_kph_roll_std_3d",
]


def main() -> None:
    """Create and save the shared chronological split."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{INPUT_FILE}"
        )

    print("Loading feature-engineered dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset shape: {df.shape}")

    # Validate required columns:

    required_columns = [
        "datetime",
        *TARGET_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # Convert and validate datetime:

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    invalid_dates = int(
        df["datetime"].isna().sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} rows contain invalid datetime values."
        )

    # Sort the data from oldest to newest.
    # location_name provides a stable secondary ordering.

    sort_columns = ["datetime"]

    if "location_name" in df.columns:
        sort_columns.append("location_name")

    df = df.sort_values(
        by=sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    # Separate predictors and targets:

    y = df[TARGET_COLUMNS].copy()

    columns_to_remove = [
        *TARGET_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]

    existing_removals = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]


    # Create a strict chronological 80/20 split:
# ---------------------------------------------
# Create chronological train/test split
# independently for each country
# ---------------------------------------------

    train_parts = []
    test_parts = []

    for country, country_df in df.groupby("country", sort=False):

        country_df = (
        country_df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

        split_index = int(
        len(country_df) * TRAIN_RATIO
    )

        split_index = min(
        max(split_index, 1),
        len(country_df) - 1,
    )

        cutoff_datetime = country_df.loc[
        split_index,
        "datetime"
    ]

    # Keep all rows with the cutoff timestamp in the test set.
    # This prevents the same timestamp from appearing in both partitions.

        test_country = country_df[
        country_df["datetime"] >= cutoff_datetime
    ]
        train_parts.append(train_country)
        test_parts.append(test_country)

    train_df = (
    pd.concat(train_parts)
    .reset_index(drop=True)
)

    test_df = (
    pd.concat(test_parts)
    .reset_index(drop=True)
)

    # Separate targets and predictors:

    y = df[TARGET_COLUMNS].copy()

    columns_to_remove = [
        *TARGET_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]

    existing_removals = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]
    X = df.drop( columns=existing_removals ).copy()
    X_train = train_df.drop(
    columns=existing_removals
)

    X_test = test_df.drop(
    columns=existing_removals
)
    y_train = train_df[TARGET_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMNS].copy()
   

    # Validate the split:

    if X_train.empty:
        raise ValueError(
            "The training dataset is empty."
        )

    if X_test.empty:
        raise ValueError(
            "The testing dataset is empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training predictors and targets have different row counts."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing predictors and targets have different row counts."
        )

    if X_train.isnull().any().any():
        raise ValueError(
            "X_train contains missing values."
        )

    if X_test.isnull().any().any():
        raise ValueError(
            "X_test contains missing values."
        )

    if y_train.isnull().any().any():
        raise ValueError(
            "y_train contains missing values."
        )

    if y_test.isnull().any().any():
        raise ValueError(
            "y_test contains missing values."
        )

    train_start = pd.to_datetime(
        X_train["datetime"]
    ).min()

    train_end = pd.to_datetime(
        X_train["datetime"]
    ).max()

    test_start = pd.to_datetime(
        X_test["datetime"]
    ).min()

    test_end = pd.to_datetime(
        X_test["datetime"]
    ).max()

    timestamp_overlap = (
        train_end >= test_start
    )

    

    # Save outputs:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train_path = (
        OUTPUT_DIR / "X_train.csv"
    )

    X_test_path = (
        OUTPUT_DIR / "X_test.csv"
    )

    y_train_path = (
        OUTPUT_DIR / "y_train.csv"
    )

    y_test_path = (
        OUTPUT_DIR / "y_test.csv"
    )

    metadata_path = (
        OUTPUT_DIR / "split_metadata.json"
    )

    X_train.to_csv(
        X_train_path,
        index=False,
    )

    X_test.to_csv(
        X_test_path,
        index=False,
    )

    y_train.to_csv(
        y_train_path,
        index=False,
    )

    y_test.to_csv(
        y_test_path,
        index=False,
    )

    # Save metadata describing the shared split:

    metadata = {
        "source_file": str(INPUT_FILE),
        "split_method": "strict_chronological",
        "requested_train_ratio": TRAIN_RATIO,
        "actual_train_ratio": (
            len(X_train) / len(df)
        ),
        "actual_test_ratio": (
            len(X_test) / len(df)
        ),
        "cutoff_datetime": str(
            cutoff_datetime
        ),
        "training_rows": int(
            len(X_train)
        ),
        "testing_rows": int(
            len(X_test)
        ),
        "feature_count": int(
            X.shape[1]
        ),
        "feature_columns": list(
            X.columns
        ),
        "target_columns": TARGET_COLUMNS,
        "leakage_columns_removed": existing_removals,
        "training_start": str(
            train_start
        ),
        "training_end": str(
            train_end
        ),
        "testing_start": str(
            test_start
        ),
        "testing_end": str(
            test_end
        ),
        "timestamp_overlap": bool(
            timestamp_overlap
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    # Display a console summary:

    print("\nShared split created successfully.")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    print("\nChronological boundaries:")
    print(f"Training starts: {train_start}")
    print(f"Training ends:   {train_end}")
    print(f"Testing starts:  {test_start}")
    print(f"Testing ends:    {test_end}")
    print(f"Timestamp overlap: {timestamp_overlap}")

    print("\nTarget columns:")
    for target in TARGET_COLUMNS:
        print(f"  - {target}")

    print("\nLeakage columns removed:")
    for column in existing_removals:
        print(f"  - {column}")

    print("\nFiles saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
