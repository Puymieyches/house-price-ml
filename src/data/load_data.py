"""
Data loading utilities for the house price prediction project.

This module handles downloading, caching, and initial loading of
housing datasets from various sources.

"""

import zipfile
import os
from pathlib import Path
from typing import Tuple, Optional
import requests
import pandas as pd
import numpy as np
from src.utils.logging import setup_logging

logger = setup_logging(__name__)

def download_housing_data(force_download: bool = False) -> None:
    """
    Download housing dataset if not already present.
    
    Args:
        force_download: If True, redownload even if file exists
    """
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_dir / "housing_data.csv"
    
    if dataset_path.exists() and not force_download:
        logger.info("Dataset already exists at %s", dataset_path)
        return
    
    logger.info("Downloading housing dataset...")
    
    # For this example, we'll create a realistic synthetic dataset
    # In practice, you'd download from a real source
    housing_data = generate_realistic_housing_data(n_samples=10000)
    
    # Save to CSV
    housing_data.to_csv(dataset_path, index=False)
    logger.info("Dataset saved to %s", dataset_path)
    
    # Create data dictionary
    create_data_dictionary(data_dir)

def generate_realistic_housing_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate realistic synthetic housing data for learning purposes.
    
    This creates data that mimics real housing markets with:
    - Realistic price distributions
    - Correlated features (size -> price)
    - Some missing values and outliers
    - Different neighborhoods with different characteristics
    """
    np.random.seed(42)  # For reproducibility
    
    # Define neighborhoods with different characteristics
    neighborhoods = {
        'Downtown': {'base_price': 450000, 'price_per_sqft': 280, 'premium': 1.3},
        'Suburbs_North': {'base_price': 320000, 'price_per_sqft': 180, 'premium': 1.0},
        'Suburbs_South': {'base_price': 280000, 'price_per_sqft': 160, 'premium': 0.9},
        'Waterfront': {'base_price': 650000, 'price_per_sqft': 350, 'premium': 1.8},
        'Historic': {'base_price': 380000, 'price_per_sqft': 220, 'premium': 1.2},
        'University': {'base_price': 250000, 'price_per_sqft': 140, 'premium': 0.8},
        'Industrial': {'base_price': 200000, 'price_per_sqft': 110, 'premium': 0.7}
    }
    
    data = []
    
    for i in range(n_samples):
        # Select neighborhood
        neighborhood = np.random.choice(list(neighborhoods.keys()))
        neighborhood_info = neighborhoods[neighborhood]
        
        # Generate basic features
        square_feet = np.random.normal(1800, 600)
        square_feet = max(500, square_feet)  # Minimum size
        
        # Bedrooms roughly correlated with size
        bedrooms = max(1, int(np.random.normal(square_feet / 400, 1)))
        bedrooms = min(bedrooms, 8)  # Maximum bedrooms
        
        # Bathrooms correlated with bedrooms
        bathrooms = max(1, np.random.normal(bedrooms * 0.75, 0.5))
        bathrooms = round(bathrooms * 2) / 2  # Round to nearest 0.5
        
        # Age of house
        age = np.random.exponential(15)  # Most houses are newer
        age = min(age, 150)  # Cap at 150 years
        
        # Lot size
        lot_size = np.random.lognormal(8.5, 0.8)  # Log-normal distribution
        lot_size = max(lot_size, 1000)  # Minimum lot size
        
        # Property type
        property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi-Family']
        property_weights = [0.6, 0.2, 0.15, 0.05]
        property_type = np.random.choice(property_types, p=property_weights)
        
        # Condition (1-5 scale)
        condition = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.5, 0.25, 0.05])
        
        # Features that affect price
        has_garage = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])  # 0, 1, or 2 car garage
        has_pool = np.random.choice([0, 1], p=[0.85, 0.15])
        has_fireplace = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Calculate base price using neighborhood and size
        base_price = (
            neighborhood_info['base_price'] + 
            square_feet * neighborhood_info['price_per_sqft']
        )
        
        # Apply adjustments
        price_multiplier = 1.0
        
        # Age adjustment
        if age < 5:
            price_multiplier *= 1.1  # New construction premium
        elif age > 50:
            price_multiplier *= 0.9  # Older home discount
            
        # Condition adjustment
        condition_multipliers = {1: 0.7, 2: 0.85, 3: 1.0, 4: 1.15, 5: 1.3}
        price_multiplier *= condition_multipliers[condition]
        
        # Feature adjustments
        if has_garage == 1:
            price_multiplier *= 1.05
        elif has_garage == 2:
            price_multiplier *= 1.1
            
        if has_pool:
            price_multiplier *= 1.08
            
        if has_fireplace:
            price_multiplier *= 1.03
            
        # Property type adjustment
        type_multipliers = {
            'Single Family': 1.0,
            'Condo': 0.85,
            'Townhouse': 0.92,
            'Multi-Family': 1.1
        }
        price_multiplier *= type_multipliers[property_type]
        
        # Calculate final price with some noise
        final_price = base_price * price_multiplier
        final_price *= np.random.normal(1.0, 0.1)  # Add 10% random variation
        final_price = max(final_price, 50000)  # Minimum price
        
        # Sale date (random over past 3 years)
        days_ago = np.random.randint(0, 1095)  # 3 years
        sale_date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
        
        # Days on market
        days_on_market = max(1, int(np.random.exponential(30)))
        
        data.append({
            'price': round(final_price, 0),
            'square_feet': round(square_feet, 0),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'lot_size': round(lot_size, 0),
            'age': round(age, 0),
            'neighborhood': neighborhood,
            'property_type': property_type,
            'condition': condition,
            'garage': has_garage,
            'pool': has_pool,
            'fireplace': has_fireplace,
            'sale_date': sale_date.strftime('%Y-%m-%d'),
            'days_on_market': days_on_market
        })
    
    df = pd.DataFrame(data)
    
    # Introduce some realistic missing values
    # Missing lot_size for condos (common in real estate data)
    condo_mask = df['property_type'] == 'Condo'
    df.loc[condo_mask & (np.random.random(condo_mask.sum()) < 0.8), 'lot_size'] = np.nan
    
    # Some missing garage info
    df.loc[np.random.random(len(df)) < 0.02, 'garage'] = np.nan
    
    # Some missing condition ratings
    df.loc[np.random.random(len(df)) < 0.05, 'condition'] = np.nan
    
    # Add some outliers (data entry errors)
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.001), replace=False)
    for idx in outlier_indices:
        if np.random.random() < 0.5:
            # Price outlier
            df.loc[idx, 'price'] *= np.random.choice([0.1, 10])  # Very cheap or very expensive
        else:
            # Size outlier
            df.loc[idx, 'square_feet'] *= np.random.choice([0.1, 5])  # Very small or very large
    
    return df

def create_data_dictionary(data_dir: Path) -> None:
    """Create a data dictionary explaining each column."""
    
    dictionary = """
# Housing Dataset Data Dictionary

## Overview
This dataset contains information about residential property sales, including property characteristics, location, and sale details.

## Columns

### Target Variable
- **price**: Sale price in USD (target variable to predict)

### Property Characteristics  
- **square_feet**: Living area in square feet
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms (can be fractional, e.g., 2.5)
- **lot_size**: Lot size in square feet
- **age**: Age of the property in years
- **property_type**: Type of property (Single Family, Condo, Townhouse, Multi-Family)
- **condition**: Property condition on 1-5 scale (1=Poor, 5=Excellent)

### Location
- **neighborhood**: Neighborhood where property is located

### Features/Amenities
- **garage**: Number of garage spaces (0, 1, or 2)
- **pool**: Has swimming pool (0=No, 1=Yes)  
- **fireplace**: Has fireplace (0=No, 1=Yes)

### Sale Information
- **sale_date**: Date of sale (YYYY-MM-DD format)
- **days_on_market**: Number of days property was listed before sale

## Data Quality Notes
- Some condos may have missing lot_size (common for high-rise buildings)
- Small percentage of missing values in garage and condition columns
- Dataset may contain some outliers due to data entry errors or unique properties
- All prices are in nominal dollars (not adjusted for inflation)

## Neighborhoods
- **Downtown**: Urban core, high prices, walkable
- **Suburbs_North**: Family-friendly suburban area  
- **Suburbs_South**: More affordable suburban area
- **Waterfront**: Premium location with water access
- **Historic**: Historic district with character homes
- **University**: Near university campus, mixed housing types
- **Industrial**: Lower-cost area near industrial zones
"""
    
    dict_path = data_dir / "data_dictionary.md"
    with open(dict_path, 'w') as f:
        f.write(dictionary)
    
    logger.info("Data dictionary created at %s", dict_path)

def load_housing_data() -> pd.DataFrame:
    """
    Load the housing dataset with basic preprocessing.
    
    Returns:
        pd.DataFrame: Raw housing data
    """
    data_path = Path("data/raw/housing_data.csv")
    
    if not data_path.exists():
        logger.info("Housing data not found, downloading...")
        download_housing_data()
    
    logger.info("Loading housing data from %s", data_path)
    df = pd.read_csv(data_path)
    
    # Convert sale_date to datetime
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    logger.info("Loaded %d records with %d columns", len(df), len(df.columns))
    
    return df

if __name__ == "__main__":
    # Download data when script is run directly
    download_housing_data(force_download=True)
    print("Housing data downloaded successfully!")
