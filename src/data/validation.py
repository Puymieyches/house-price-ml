"""
Data validation utilities for housing price prediction.

This module provides comprehensive data quality checks and validation
rules specific to real estate data.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.utils.logging import setup_logging

logger = setup_logging(__name__)

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    
class DataValidator:
    """Comprehensive data validation for housing data."""
    
    def __init__(self):
        self.rules = self._define_validation_rules()
        self.results = []
    
    def _define_validation_rules(self) -> List[ValidationRule]:
        """Define validation rules for housing data."""
        return [
            ValidationRule("price_range", "Price should be between $10K and $50M", "error"),
            ValidationRule("sqft_range", "Square feet should be between 100 and 50,000", "error"),
            ValidationRule("bedroom_range", "Bedrooms should be between 0 and 20", "warning"),
            ValidationRule("bathroom_range", "Bathrooms should be between 0.5 and 20", "warning"),
            ValidationRule("age_range", "Age should be between 0 and 300 years", "warning"),
            ValidationRule("missing_critical", "Critical fields shouldn't be missing", "error"),
            ValidationRule("logical_consistency", "Bedrooms/bathrooms should match property size", "warning"),
            ValidationRule("duplicate_records", "No duplicate property records allowed", "error"),
        ]
    
    def validate_price_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate price values are within reasonable range."""
        price_issues = df[(df['price'] < 10000) | (df['price'] > 50000000)]
        
        return {
            'rule_name': 'price_range',
            'passed': len(price_issues) == 0,
            'issue_count': len(price_issues),
            'details': f"{len(price_issues)} properties with unrealistic prices",
            'severity': 'error' if len(price_issues) > 0 else 'info'
        }
    
    def validate_sqft_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate square footage is reasonable."""
        sqft_issues = df[(df['square_feet'] < 100) | (df['square_feet'] > 50000)]
        
        return {
            'rule_name': 'sqft_range', 
            'passed': len(sqft_issues) == 0,
            'issue_count': len(sqft_issues),
            'details': f"{len(sqft_issues)} properties with unrealistic square footage",
            'severity': 'error' if len(sqft_issues) > 0 else 'info'
        }
    
    def validate_missing_critical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in critical columns."""
        critical_cols = ['price', 'square_feet', 'neighborhood']
        missing_critical = df[critical_cols].isnull().any(axis=1).sum()
        
        return {
            'rule_name': 'missing_critical',
            'passed': missing_critical == 0,
            'issue_count': missing_critical,
            'details': f"{missing_critical} records missing critical information",
            'severity': 'error' if missing_critical > 0 else 'info'
        }
    
    def validate_logical_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for logical inconsistencies in the data."""
        issues = []
        
        # Check if bathrooms > bedrooms by a lot (might indicate data error)
        bathroom_issues = df[df['bathrooms'] > df['bedrooms'] * 2]
        if len(bathroom_issues) > 0:
            issues.append(f"{len(bathroom_issues)} properties with excessive bathrooms")
        
        # Check if square feet per bedroom is unreasonably small
        df['sqft_per_bedroom'] = df['square_feet'] / df['bedrooms'].replace(0, 1)
        small_rooms = df[df['sqft_per_bedroom'] < 50]
        if len(small_rooms) > 0:
            issues.append(f"{len(small_rooms)} properties with tiny bedrooms")
        
        return {
            'rule_name': 'logical_consistency',
            'passed': len(issues) == 0,
            'issue_count': len(issues),
            'details': '; '.join(issues) if issues else "No logical inconsistencies found",
            'severity': 'warning' if len(issues) > 0 else 'info'
        }
    
    def validate_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records."""
        # Consider records duplicates if they have same address and sale_date
        # For this synthetic data, we'll check for exact duplicates
        duplicates = df.duplicated().sum()
        
        return {
            'rule_name': 'duplicate_records',
            'passed': duplicates == 0,
            'issue_count': duplicates,
            'details': f"{duplicates} duplicate records found",
            'severity': 'error' if duplicates > 0 else 'info'
        }
    
    def run_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report."""
        
        logger.info("Starting data validation...")
        
        validation_results = []
        
        # Run each validation check
        validation_results.append(self.validate_price_range(df))
        validation_results.append(self.validate_sqft_range(df))
        validation_results.append(self.validate_missing_critical(df))
        validation_results.append(self.validate_logical_consistency(df))
        validation_results.append(self.validate_duplicates(df))
        
        # Summarize results
        errors = [r for r in validation_results if r['severity'] == 'error' and not r['passed']]
        warnings = [r for r in validation_results if r['severity'] == 'warning' and not r['passed']]
        
        validation_summary = {
            'total_checks': len(validation_results),
            'passed_checks': len([r for r in validation_results if r['passed']]),
            'errors': len(errors),
            'warnings': len(warnings),
            'validation_results': validation_results,
            'overall_status': 'PASS' if len(errors) == 0 else 'FAIL'
        }
        
        # Log summary
        logger.info("Validation complete: %s", validation_summary['overall_status'])
        logger.info("Checks passed: %s/%s", validation_summary['passed_checks'], validation_summary['total_checks'])
        if errors:
            logger.error("Found %d critical errors", len(errors))
        if warnings:
            logger.warning("Found %d warnings", len(warnings))
        
        return validation_summary

# Usage example
def run_data_validation_example():
    """Example of how to use the data validation framework."""
    
    # Load data
    from src.data.load_data import load_housing_data
    df = load_housing_data()
    
    # Run validation
    validator = DataValidator()
    results = validator.run_validation(df)
    
    # Print results
    print("🔍 DATA VALIDATION RESULTS")
    print("=" * 40)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Checks Passed: {results['passed_checks']}/{results['total_checks']}")
    print(f"Errors: {results['errors']}")
    print(f"Warnings: {results['warnings']}")
    
    print("\n📋 Detailed Results:")
    for result in results['validation_results']:
        status = "✅" if result['passed'] else "❌" if result['severity'] == 'error' else "⚠️"
        print(f"{status} {result['rule_name']}: {result['details']}")
        