#!/usr/bin/env python3

"""
Data Inspection Script

This script examines the structure of the data files to help debug the ML pipeline issues.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()

def inspect_excel_file(file_path):
    """Inspect an Excel file and print its structure"""
    print(f"Examining Excel file: {file_path}")
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Sample data (first 5 rows):")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def inspect_csv_file(file_path):
    """Inspect a CSV file and print its structure"""
    print(f"Examining CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Sample data (first 5 rows):")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def main():
    """Main function to inspect all data files"""
    print("Data Inspection Report")
    print("=====================")
    
    # Inspect demographic data
    demo_data = inspect_excel_file(PROJECT_DIR / "age_gender.xlsx")
    
    # Inspect subcortical volumes
    subcort_data = inspect_csv_file(PROJECT_DIR / "stats" / "all_subcortical_volumes.csv")
    
    # Inspect cortical thickness data
    thickness_data = inspect_csv_file(PROJECT_DIR / "thickness_output" / "all_subjects_regional_thickness.csv")
    
    # Check unique groups in subcortical data
    if subcort_data is not None:
        print("\nUnique Groups in Subcortical Data:")
        print(subcort_data['group'].unique())
    
    # Check subject overlap
    if subcort_data is not None and thickness_data is not None:
        subcort_subjects = subcort_data['subject_id'].unique()
        thickness_subjects = thickness_data['Subject'].unique() if 'Subject' in thickness_data.columns else []
        
        print(f"\nSubcortical data has {len(subcort_subjects)} unique subjects")
        print(f"Cortical thickness data has {len(thickness_subjects)} unique subjects")
        
        common_subjects = set(subcort_subjects).intersection(set(thickness_subjects))
        print(f"Common subjects between datasets: {len(common_subjects)}")

if __name__ == "__main__":
    main()