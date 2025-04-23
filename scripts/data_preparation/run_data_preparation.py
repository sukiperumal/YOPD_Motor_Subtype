#!/usr/bin/env python3

"""
Run Data Preparation for YOPD Motor Subtype Analysis

This script serves as the main entry point for generating the neuroimaging
features (subcortical volumes and cortical thickness) needed for the 
machine learning classifier.

It orchestrates:
1. Extracting features from preprocessed MRI data
2. Generating and saving CSV files with subcortical and cortical measurements
3. Creating visualizations of the extracted data
4. Preparing the data in a format suitable for the ML pipeline
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Set up project paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
LOG_DIR = PROJECT_DIR / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
log_filename = f'data_preparation_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / log_filename)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('data_preparation')

def import_and_run_feature_extraction(args):
    """Import and run the feature extraction module"""
    try:
        logger.info("Importing feature extraction module")
        sys.path.append(str(PROJECT_DIR / "scripts" / "data_preparation"))
        
        # Import the feature extraction module
        from generate_neuroimaging_features import process_all_subjects
        
        # Run feature extraction
        logger.info("Running feature extraction")
        process_all_subjects()
        
        logger.info("Feature extraction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running feature extraction: {e}")
        return False

def verify_data_files():
    """Verify that the necessary data files have been generated"""
    try:
        logger.info("Verifying generated data files")
        
        # Check subcortical volumes file
        subcort_file = PROJECT_DIR / "stats" / "all_subcortical_volumes.csv"
        cortical_file = PROJECT_DIR / "thickness_output" / "all_subjects_regional_thickness.csv"
        
        subcort_exists = subcort_file.exists()
        cortical_exists = cortical_file.exists()
        
        if subcort_exists:
            logger.info(f"Subcortical volume data exists: {subcort_file}")
        else:
            logger.warning(f"Subcortical volume data not found: {subcort_file}")
            
        if cortical_exists:
            logger.info(f"Cortical thickness data exists: {cortical_file}")
        else:
            logger.warning(f"Cortical thickness data not found: {cortical_file}")
            
        # Return overall status
        return subcort_exists and cortical_exists
    except Exception as e:
        logger.error(f"Error verifying data files: {e}")
        return False

def prepare_data_for_ml():
    """Prepare data for ML pipeline by adding required columns"""
    try:
        logger.info("Preparing data for ML pipeline")
        
        # Import pandas
        import pandas as pd
        
        # Load and process subcortical data
        subcort_file = PROJECT_DIR / "stats" / "all_subcortical_volumes.csv"
        if subcort_file.exists():
            logger.info(f"Processing {subcort_file}")
            df = pd.read_csv(subcort_file)
            
            # Check if required columns exist
            if 'subject_id' not in df.columns:
                logger.error("Required column 'subject_id' not found in subcortical data")
                return False
                
            # Ensure data is in the right format for ML
            if 'group' not in df.columns:
                logger.warning("Group column not found in subcortical data, will be added from demographics")
                
            # Save processed data
            df.to_csv(subcort_file, index=False)
            logger.info(f"Processed subcortical data saved: {len(df)} rows")
        
        # Load and process cortical data
        cortical_file = PROJECT_DIR / "thickness_output" / "all_subjects_regional_thickness.csv"
        if cortical_file.exists():
            logger.info(f"Processing {cortical_file}")
            df = pd.read_csv(cortical_file)
            
            # Check if required columns exist
            if 'Subject' not in df.columns:
                logger.error("Required column 'Subject' not found in cortical data")
                return False
                
            # Add subject_id column if it doesn't exist (ML pipeline expects this)
            if 'subject_id' not in df.columns:
                logger.info("Adding subject_id column to cortical data")
                df['subject_id'] = df['Subject']
                
            # Save processed data
            df.to_csv(cortical_file, index=False)
            logger.info(f"Processed cortical data saved: {len(df)} rows")
            
        # Create a data verification file
        with open(PROJECT_DIR / "data_verification_output.txt", 'w') as f:
            f.write(f"Data preparation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Subcortical data: {'Found' if subcort_file.exists() else 'Not found'}\n")
            f.write(f"Cortical data: {'Found' if cortical_file.exists() else 'Not found'}\n")
            
        return True
    except Exception as e:
        logger.error(f"Error preparing data for ML: {e}")
        return False

def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run data preparation for YOPD Motor Subtype Analysis')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip feature extraction, use existing data files')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing data files without extraction')
    args = parser.parse_args()
    
    logger.info("Starting data preparation for YOPD Motor Subtype Analysis")
    
    # Run the feature extraction process
    if args.verify_only:
        logger.info("Verification mode only, skipping feature extraction")
        extraction_success = True
    elif args.skip_extraction:
        logger.info("Skipping feature extraction as requested")
        extraction_success = True
    else:
        logger.info("Starting feature extraction process")
        extraction_success = import_and_run_feature_extraction(args)
        
    # Verify that the necessary data files have been generated
    verification_success = verify_data_files()
    
    # Prepare data for ML if previous steps were successful
    if extraction_success and verification_success:
        logger.info("Data files verified successfully, preparing for ML pipeline")
        prep_success = prepare_data_for_ml()
        
        if prep_success:
            logger.info("Data preparation completed successfully")
            logger.info("Ready to run ML analysis")
        else:
            logger.error("Data preparation for ML failed")
    else:
        if not extraction_success:
            logger.error("Feature extraction failed")
        if not verification_success:
            logger.error("Data file verification failed")
            
    logger.info("Data preparation process completed")

if __name__ == "__main__":
    main()