#!/usr/bin/env python3

"""
create_test_data.py

This script creates sample NIFTI files for testing the resting-state fMRI pipeline.
It generates a small number of 4D NIFTI files with random data, simulating
resting-state fMRI data, which can be used to test the pipeline functionality.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('sample_data')

# Set up paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
SAMPLE_DIR = PROJECT_DIR / "sample_data"
FMRI_PROCESSED_DIR = PROJECT_DIR / "fmri_processed"

# Create sample directories
os.makedirs(SAMPLE_DIR, exist_ok=True)

def load_subject_info():
    """Load subject information from all_subjects.csv"""
    try:
        subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
        logger.info(f"Loaded {len(subjects_df)} subjects from all_subjects.csv")
        return subjects_df
    except FileNotFoundError:
        logger.error("Subject information file not found")
        sys.exit(1)

def create_sample_data():
    """Create sample 4D NIFTI files for testing"""
    subjects_df = load_subject_info()
    
    # Select a sample of subjects from each group
    groups = subjects_df['group'].unique()
    sample_subjects = []
    
    for group in groups:
        # Select 2 subjects from each group
        group_subjects = subjects_df[subjects_df['group'] == group]['subject_id'].tolist()
        if len(group_subjects) >= 2:
            sample_subjects.extend(group_subjects[:2])
        else:
            sample_subjects.extend(group_subjects)
    
    logger.info(f"Creating sample data for {len(sample_subjects)} subjects: {', '.join(sample_subjects)}")
    
    # Create sample 4D NIFTI files
    for subject_id in sample_subjects:
        # Create subject directory
        subject_dir = FMRI_PROCESSED_DIR / subject_id
        os.makedirs(subject_dir, exist_ok=True)
        
        # Create a 4D NIFTI file (64x64x64x100 - simulating 100 timepoints)
        data = np.random.random((64, 64, 64, 100))
        affine = np.eye(4)  # Identity matrix for affine transform
        img = nib.Nifti1Image(data, affine)
        
        # Save as func_preprocessed.nii.gz
        output_file = subject_dir / 'func_preprocessed.nii.gz'
        nib.save(img, str(output_file))
        logger.info(f"Created sample file: {output_file}")
    
    logger.info(f"Sample data creation complete. Created data for {len(sample_subjects)} subjects.")
    return sample_subjects

if __name__ == "__main__":
    # Check if subject directories exist
    if not FMRI_PROCESSED_DIR.exists():
        os.makedirs(FMRI_PROCESSED_DIR, exist_ok=True)
        logger.info(f"Created directory: {FMRI_PROCESSED_DIR}")
    
    # Create sample data
    sample_subjects = create_sample_data()
    
    print("\nSample Data Creation Summary:")
    print(f"- Created sample data for {len(sample_subjects)} subjects")
    print(f"- Data location: {FMRI_PROCESSED_DIR}")
    print("\nYou can now run the resting-state fMRI pipeline with this sample data:")
    print("python scripts/run_rsfmri_analysis.py --skip-preprocessing")