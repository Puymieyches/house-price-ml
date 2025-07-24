"""Tests for data validation and preprocessing functions."""
import pandas as pd
from pandas.api.types import CategoricalDtype
from src.data.preprocess import (
    clean_data,
    handle_missing_values,
    remove_outliers,
    fix_data_types,
    validate_data,
    preprocess_housing_data,
)


def test_clean_data_removes_nulls():
    """Test that clean_data function removes null values."""
    # Arrange
    dirty_data = pd.DataFrame(
        {
            "price": [100000, None, 200000],
            "square_feet": [1000, 1200, None],
            "bedrooms": [2, 3, 2],
        }
    )

    # Act
    clean = clean_data(dirty_data, remove_nulls=True)

    # Assert
    assert clean.isnull().sum().sum() == 0
    assert len(clean) == 1  # Only one complete row


def test_clean_data_preserves_valid_data():
    """Test that clean_data preserves valid data."""
    # Arrange
    valid_data = pd.DataFrame(
        {
            "price": [100000, 200000, 300000],
            "square_feet": [1000, 1200, 1500],
            "bedrooms": [2, 3, 4],
        }
    )

    # Act
    clean = clean_data(valid_data, remove_nulls=True)

    # Assert
    pd.testing.assert_frame_equal(clean, valid_data)


def test_remove_outliers_iqr():
    """Test outlier removal using IQR method."""
    # Arrange
    data = pd.DataFrame(
        {
            "price": [100000, 110000, 120000, 1000000],  # Last one is outlier
            "square_feet": [1000, 1100, 1200, 1150],
        }
    )

    # Act
    result = remove_outliers(data, method="iqr")

    # Assert
    assert len(result) == 3  # Outlier removed
    assert result["price"].max() == 120000  # No more outlier


def test_fix_data_types():
    """Test data type fixing."""
    # Arrange
    data = pd.DataFrame(
        {
            "price": ["100000", "200000", "300000"],  # String prices
            "sale_date": ["2023-01-01", "2023-02-01", "2023-03-01"],  # String dates
            "neighborhood": ["Downtown", "Suburbs", "Downtown"],
        }
    )

    # Act
    result = fix_data_types(data)

    # Assert
    assert pd.api.types.is_numeric_dtype(result["price"])
    assert pd.api.types.is_datetime64_any_dtype(result["sale_date"])
    assert isinstance(result["neighborhood"].dtype, CategoricalDtype)


def test_handle_missing_values_median():
    """Test missing value imputation with median strategy."""
    # Arrange
    data = pd.DataFrame(
        {
            "price": [100000, 200000, 300000, None],
            "square_feet": [1000, None, 1500, 1200],
        }
    )

    strategy = {"price": "median", "square_feet": "median"}

    # Act
    result = handle_missing_values(data, strategy)

    # Assert
    assert result.isnull().sum().sum() == 0
    assert result["price"].iloc[3] == 200000  # Median of [100000, 200000, 300000]


def test_validate_data():
    """Test data validation function."""
    # Arrange
    data = pd.DataFrame(
        {
            "price": [100000, 50, 200000],  # One unrealistically cheap
            "square_feet": [1000, 1200, 50],  # One unrealistically small
            "bedrooms": [2, 3, 2],
        }
    )

    # Act
    validation = validate_data(data)

    # Assert
    assert validation["total_records"] == 3
    assert len(validation["issues"]) > 0  # Should flag issues


def test_preprocess_housing_data():
    """Test complete preprocessing pipeline."""
    # Arrange
    raw_data = pd.DataFrame(
        {
            "price": ["100000", None, "200000"],
            "square_feet": [1000, 1200, None],
            "sale_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "neighborhood": ["Downtown", "Suburbs", "Downtown"],
        }
    )

    # Act
    result = preprocess_housing_data(raw_data)

    # Assert
    assert len(result) > 0  # Should have some data left
    assert pd.api.types.is_numeric_dtype(result["price"])
    assert pd.api.types.is_datetime64_any_dtype(result["sale_date"])
