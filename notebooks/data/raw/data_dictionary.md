
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
