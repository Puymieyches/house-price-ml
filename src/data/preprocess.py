"""
Data preprocessing utilities for housing price prediction.

This module provides functions for cleaning and preprocessing housing data,
including handling missing values, outliers, and data type conversions.

"""
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats


def clean_data(
    df: pd.DataFrame,
    remove_nulls: bool = True,
    handle_outliers: bool = False,
    outlier_method: str = "iqr",
) -> pd.DataFrame:
    """
    Clean housing data by handling missing values and outliers.

    Args:
        df (pd.DataFrame): Raw housing data
        remove_nulls (bool): Whether to remove rows with null values
        handle_outliers (bool): Whether to remove/cap outliers
        outlier_method (str): Method for outlier detection ('iqr' or 'zscore')

    Returns:
        pd.DataFrame: Cleaned housing data

    Example:
        >>> raw_data = pd.read_csv('housing.csv')
        >>> clean = clean_data(raw_data, remove_nulls=True)
        >>> print(f"Cleaned {len(raw_data)} -> {len(clean)} records")
    """

    print(f"Starting data cleaning for {len(df)} records")
    df_clean = df.copy()

    # Handle missing values
    if remove_nulls:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing values")

    # Handle outliers if requested
    if handle_outliers:
        df_clean = remove_outliers(df_clean, method=outlier_method)

    # Basic data type corrections
    df_clean = fix_data_types(df_clean)

    print(f"Data cleaning complete: {len(df_clean)} records remaining")
    return df_clean


def remove_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Remove outliers from numerical columns.

    Args:
        df (pd.DataFrame): Input data
        method (str): 'iqr' or 'zscore'

    Returns:
        pd.DataFrame: Data with outliers removed
    """

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_no_outliers = df.copy()

    for col in numerical_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":

            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask[df[col].dropna().index] = z_scores > 3

        else:
            raise ValueError(f"Unknown outlier method: {method}")

        outliers_removed = outlier_mask.sum()
        if outliers_removed > 0:
            df_no_outliers = df_no_outliers[~outlier_mask]
            print(f"Removed {outliers_removed} outliers from {col}")

    return df_no_outliers


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common data type issues in housing data.

    Args:
        df (pd.DataFrame): Input data

    Returns:
        pd.DataFrame: Data with corrected types
    """

    df_fixed = df.copy()

    # Convert sale_date to datetime if it exists and isn't already datetime
    if "sale_date" in df_fixed.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_fixed["sale_date"]):
            df_fixed["sale_date"] = pd.to_datetime(
                df_fixed["sale_date"], errors="coerce"
            )

    # Ensure numerical columns are actually numeric
    numeric_columns = [
        "price",
        "square_feet",
        "lot_size",
        "age",
        "bedrooms",
        "bathrooms",
        "garage",
    ]
    for col in numeric_columns:
        if col in df_fixed.columns:
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors="coerce")

    # Convert categorical columns to proper categories
    categorical_cols = ["neighborhood", "property_type"]
    for col in categorical_cols:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].astype("category")

    # Ensure boolean columns are proper booleans
    boolean_cols = ["pool", "fireplace"]
    for col in boolean_cols:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].astype(bool)

    return df_fixed


def handle_missing_values(
    df: pd.DataFrame, strategy: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Handle missing values using specified strategies for each column.

    Args:
        df (pd.DataFrame): Data with missing values
        strategy (Dict[str, str]): Strategy for each column
            Options: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'constant'

    Returns:
        pd.DataFrame: Data with missing values handled
    """

    if strategy is None:
        # Default strategies
        strategy = {
            "price": "drop",  # Critical field
            "square_feet": "median",  # Use median for size
            "lot_size": "median",
            "age": "median",
            "bedrooms": "mode",
            "bathrooms": "median",
            "garage": "mode",
            "condition": "mode",
            "neighborhood": "mode",
            "property_type": "mode",
            "pool": "constant",
            "fireplace": "constant",
        }

    df_imputed = df.copy()

    for column, method in strategy.items():
        if column in df_imputed.columns and df_imputed[column].isnull().any():

            missing_count = df_imputed[column].isnull().sum()

            if method == "drop":
                df_imputed = df_imputed.dropna(subset=[column])
                print(f"Dropped {missing_count} rows due to missing {column}")

            elif method == "mean":
                fill_value = df_imputed[column].mean()
                df_imputed[column] = df_imputed[column].fillna(fill_value)
                print(
                    f"Filled {missing_count} missing {column} values with mean: {fill_value:.2f}"
                )

            elif method == "median":
                fill_value = df_imputed[column].median()
                df_imputed[column] = df_imputed[column].fillna(fill_value)
                print(
                    f"Filled {missing_count} missing {column} values with median: {fill_value}"
                )

            elif method == "mode":
                if len(df_imputed[column].mode()) > 0:
                    fill_value = df_imputed[column].mode().iloc[0]
                else:
                    fill_value = "Unknown"
                df_imputed[column] = df_imputed[column].fillna(fill_value)
                print(
                    f"Filled {missing_count} missing {column} values with mode: {fill_value}"
                )

            elif method == "forward_fill":
                df_imputed[column] = df_imputed[column].fillna(method="ffill")
                print(f"Forward filled {missing_count} missing {column} values")

            elif method == "constant":
                fill_value = (
                    0 if df_imputed[column].dtype in ["int64", "float64"] else False
                )
                df_imputed[column] = df_imputed[column].fillna(fill_value)
                print(
                    f"Filled {missing_count} missing {column} values with constant: {fill_value}"
                )

            else:
                print(f"Unknown strategy '{method}' for column '{column}', skipping...")

    return df_imputed


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return a summary report.

    Args:
        df (pd.DataFrame): Data to validate

    Returns:
        Dict: Validation summary
    """

    validation_results = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict(),
        "issues": [],
    }

    # Check for unrealistic values
    if "price" in df.columns:
        cheap_houses = (df["price"] < 10000).sum()
        expensive_houses = (df["price"] > 10000000).sum()
        if cheap_houses > 0:
            validation_results["issues"].append(
                f"{cheap_houses} houses with price < $10,000"
            )
        if expensive_houses > 0:
            validation_results["issues"].append(
                f"{expensive_houses} houses with price > $10M"
            )

    if "square_feet" in df.columns:
        tiny_houses = (df["square_feet"] < 100).sum()
        huge_houses = (df["square_feet"] > 20000).sum()
        if tiny_houses > 0:
            validation_results["issues"].append(
                f"{tiny_houses} houses with < 100 sq ft"
            )
        if huge_houses > 0:
            validation_results["issues"].append(
                f"{huge_houses} houses with > 20,000 sq ft"
            )

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation_results["issues"].append(f"{duplicates} duplicate records")

    return validation_results


# Convenience function for common preprocessing pipeline
def preprocess_housing_data(
    df: pd.DataFrame,
    remove_outliers_bool: bool = False,
    missing_strategy: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for housing data.

    Args:
        df (pd.DataFrame): Raw housing data
        remove_outliers (bool): Whether to remove outliers
        missing_strategy (Dict): Custom missing value strategies

    Returns:
        pd.DataFrame: Preprocessed data ready for modeling
    """

    print("Starting complete preprocessing pipeline...")

    # Step 1: Fix data types
    df_processed = fix_data_types(df)

    # Step 2: Handle missing values
    df_processed = handle_missing_values(df_processed, missing_strategy)

    # Step 3: Remove outliers if requested
    if remove_outliers_bool:
        df_processed = remove_outliers(df_processed)

    # Step 4: Final validation
    validation = validate_data(df_processed)

    print("Preprocessing complete!")
    print(
        f"Final dataset: {validation['total_records']} records, {validation['total_columns']} columns"
    )
    if validation["issues"]:
        print("Remaining issues:")
        for issue in validation["issues"]:
            print(f"  - {issue}")

    return df_processed
