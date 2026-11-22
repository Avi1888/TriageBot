import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "chief_complaint",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "pain_score",
    "mental_status",
    "high_risk",
    "immediate_intervention",
    "expected_resources"
]


def load_patients(file_path="data/patients.csv"):
    """
    Load patient records from a CSV file and validate the dataset.

    Returns a list of patient dictionaries.
    """

    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load CSV file
    df = pd.read_csv(file_path)

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check for missing values in required columns
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values found:\n"
            f"{columns_with_missing_values}"
        )

    # Check for duplicate patient IDs
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs found.")

    # Validate expected resource values
    if not df["expected_resources"].isin([0, 1, 2]).all():
        raise ValueError(
            "expected_resources must contain only 0, 1, or 2."
        )

    # Validate age
    if not df["age"].between(18, 120).all():
        raise ValueError("Age must be between 18 and 120.")

    # Validate heart rate
    if not df["heart_rate"].between(0, 300).all():
        raise ValueError("Invalid heart rate values.")

    # Validate respiratory rate
    if not df["respiratory_rate"].between(0, 100).all():
        raise ValueError("Invalid respiratory rate values.")

    # Validate oxygen saturation
    if not df["oxygen_saturation"].between(0, 100).all():
        raise ValueError("Oxygen saturation must be between 0 and 100.")

    # Validate pain score
    if not df["pain_score"].between(0, 10).all():
        raise ValueError("Pain score must be between 0 and 10.")

    # Validate Boolean fields
    boolean_columns = ["high_risk", "immediate_intervention"]

    for column in boolean_columns:
        if not df[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain only True or False.")

        # Convert dataset into dictionaries
    patients = df.to_dict(orient="records")
    
    return patients
