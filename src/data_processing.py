"""
High-Paying Jobs Analysis - Data Processing Module

This module contains functions for cleaning and processing BLS and Census data
for high-paying jobs analysis.

Author: Marwa BS
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle data processing for high-paying jobs analysis.
    """
    
    def __init__(self):
        self.bls_data = None
        self.census_data = None
        self.merged_data = None
        
    def load_bls_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and perform initial cleaning of BLS data.
        
        Args:
            filepath (str): Path to BLS data file
            
        Returns:
            pd.DataFrame: Cleaned BLS data
        """
        try:
            logger.info(f"Loading BLS data from {filepath}")
            data = pd.read_excel(filepath)
            
            # Select relevant columns
            relevant_columns = [
                'AREA_TITLE', 'PRIM_STATE', 'LOC_QUOTIENT', 'OCC_CODE', 
                'OCC_TITLE', 'TOT_EMP', 'JOBS_1000', 'H_MEAN', 'A_MEAN'
            ]
            
            data = data[relevant_columns].copy()
            
            # Convert numeric columns
            numeric_columns = ['LOC_QUOTIENT', 'TOT_EMP', 'H_MEAN', 'A_MEAN', 'JOBS_1000']
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # Clean text columns
            data['AREA_TITLE'] = data['AREA_TITLE'].str.strip().str.title()
            data['OCC_TITLE'] = data['OCC_TITLE'].str.strip().str.title()
            
            # Clean occupation codes
            data['OCC_CODE'] = data['OCC_CODE'].str.replace('-', '', regex=False)
            
            # Filter for high-paying jobs (>= $100K)
            data = data[data['A_MEAN'] >= 100000]
            
            # Remove non-mainland US states
            unwanted_states = ['GU', 'PR', 'VI', 'DC']
            data = data[~data['PRIM_STATE'].isin(unwanted_states)]
            
            self.bls_data = data
            logger.info(f"BLS data loaded successfully. Shape: {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading BLS data: {str(e)}")
            raise
    
    def load_census_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and clean Census data.
        
        Args:
            filepath (str): Path to Census data file
            
        Returns:
            pd.DataFrame: Cleaned Census data
        """
        try:
            logger.info(f"Loading Census data from {filepath}")
            data = pd.read_csv(filepath)
            
            # Select relevant columns
            relevant_columns = [
                'STATEICP', 'DEGFIELDD', 'EDUCD', 'OCCSOC', 
                'INCTOT', 'SEX', 'AGE'
            ]
            
            data = data[relevant_columns].copy()
            
            # Filter for high income (>= $100K)
            data = data[data['INCTOT'] >= 100000]
            
            # Apply mappings
            data = self._apply_census_mappings(data)
            
            # Drop missing values
            data = data.dropna()
            
            self.census_data = data
            logger.info(f"Census data loaded successfully. Shape: {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading Census data: {str(e)}")
            raise
    
    def _apply_census_mappings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply various mappings to Census data."""
        
        # Gender mapping
        data['SEX'] = data['SEX'].astype(str).replace({'1': 'Male', '2': 'Female'})
        
        # State mapping (abbreviated for brevity)
        state_map = {
            1: "Connecticut", 2: "Maine", 4: "Massachusetts", 5: "New Hampshire",
            # ... (include full mapping)
            56: "Wyoming"
        }
        data['STATE'] = data['STATEICP'].map(state_map)
        
        # Education mapping
        education_map = {
            101: "Bachelor's degree",
            114: "Master's degree", 
            115: "Professional degree",
            116: "Doctoral degree"
        }
        data['EDUCATION_LABEL'] = data['EDUCD'].map(education_map)
        
        # Degree field mapping (abbreviated)
        degree_field_map = {
            1100: "Agriculture",
            2400: "Engineering",
            # ... (include full mapping)
        }
        data['DEGFIELDD_NAME'] = data['DEGFIELDD'].map(degree_field_map)
        
        # Clean occupation codes
        data['OCC_CODE'] = data['OCCSOC'].astype(str).str.replace('-', '')
        data['Annual_Income'] = data['INCTOT']
        
        return data
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge BLS and Census datasets.
        
        Returns:
            pd.DataFrame: Merged dataset
        """
        if self.bls_data is None or self.census_data is None:
            raise ValueError("Both BLS and Census data must be loaded first")
        
        # Prepare BLS data
        bls_clean = self.bls_data.rename(columns={'AREA_TITLE': 'STATE'})
        
        # Merge datasets
        merged = pd.merge(
            self.census_data, 
            bls_clean, 
            on=['OCC_CODE', 'STATE'], 
            how='inner'
        )
        
        # Rename columns for clarity
        column_mapping = {
            'PRIM_STATE': 'State Abbreviation',
            'STATE': 'State',
            'SEX': 'Gender',
            'AGE': 'Age',
            'EDUCD': 'Education Code',
            'EDUCATION_LABEL': 'Education Level',
            'DEGFIELDD': 'Degree Field',
            'OCC_CODE': 'Occupation Code',
            'OCC_TITLE': 'Occupation',
            'Annual_Income': 'Annual Income',
            'TOT_EMP': 'Employment',
            'LOC_QUOTIENT': 'Location Quotient',
            'JOBS_1000': 'Jobs per 1000',
            'H_MEAN': 'Hourly Mean',
            'A_MEAN': 'Annual Mean Wage'
        }
        
        merged = merged.rename(columns=column_mapping)
        
        self.merged_data = merged
        logger.info(f"Data merged successfully. Final shape: {merged.shape}")
        
        return merged
    
    def save_processed_data(self, filepath: str) -> None:
        """Save processed data to CSV."""
        if self.merged_data is None:
            raise ValueError("No merged data to save")
        
        self.merged_data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        Dict: Data quality summary
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return quality_report
