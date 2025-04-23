#!/usr/bin/env python3

"""
preprocess_fmri.py

This script preprocesses resting-state fMRI data for YOPD Motor Subtype Analysis:
1. Applies fieldmap-based distortion correction (TOPUP)
2. Performs motion correction (MCFLIRT)
3. Slice timing correction
4. Registration to anatomical T1 and then to MNI space
5. Spatial smoothing
6. High-pass temporal filtering

BIDS-compliant input data structure is expected.
"""

import os
import sys
import subprocess
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MCFLIRT, FLIRT, BET, FAST, SUSAN, ImageMaths, IsotropicSmooth
from nipype.interfaces.fsl.utils import ConvertXFM
from concurrent.futures import ProcessPoolExecutor
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'fmri_preprocessing_{time.strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('fmri_preprocessing')

# Set up paths - Fix for Windows compatibility
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
# Look for data either in the project directory or in the fmri_processed directory
DATASET_DIR = PROJECT_DIR  # Use project dir as base
OUTPUT_DIR = PROJECT_DIR / "fmri_processed"
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
LOG_DIR = PROJECT_DIR / "logs"

# Create output directories
for directory in [OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Set FSL environment variables (adjust for Windows)
if sys.platform == "win32":
    os.environ['FSLDIR'] = r'C:\Program Files\FSL'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
else:
    os.environ['FSLDIR'] = '/usr/local/fsl'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

# Function to load subject information
def load_subject_info():
    """Load subject information from the all_subjects.csv file"""
    try:
        subjects_df = pd.read_excel(PROJECT_DIR / "age_gender.xlsx")
        logger.info(f"Loaded {len(subjects_df)} subjects from age_gender.xlsx")
        return subjects_df
    except FileNotFoundError:
        try:
            subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
            logger.info(f"Loaded {len(subjects_df)} subjects from all_subjects.csv")
            return subjects_df
        except FileNotFoundError:
            logger.error("Subject information file not found")
            sys.exit(1)

# Function to find functional MRI data
def find_func_data(subject_id, group):
    """Find functional MRI data for a given subject"""
    # Define possible paths based on BIDS structure - look in multiple locations
    potential_dirs = [
        DATASET_DIR / "data" / group / subject_id / "ses-01" / "func",
        DATASET_DIR / "data" / subject_id / "ses-01" / "func",
        DATASET_DIR / "data" / group / subject_id / "func",
        DATASET_DIR / "data" / subject_id / "func",
        DATASET_DIR / "BIDS" / group / subject_id / "ses-01" / "func",
        DATASET_DIR / "BIDS" / subject_id / "ses-01" / "func",
        DATASET_DIR / "BIDS" / group / subject_id / "func",
        DATASET_DIR / "BIDS" / subject_id / "func",
        OUTPUT_DIR / subject_id,  # Check if there's already processed data
        # Add each path with 'sub-' prefix for BIDS compliance
        DATASET_DIR / "data" / group / f"sub-{subject_id}" / "ses-01" / "func",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "ses-01" / "func",
        DATASET_DIR / "data" / group / f"sub-{subject_id}" / "func",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "func",
        DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "ses-01" / "func",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "ses-01" / "func",
        DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "func",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "func",
    ]
    
    # Check if the subject ID already has the "sub-" prefix
    if not subject_id.startswith("sub-"):
        # Also check paths with the original subject ID (without adding "sub-")
        additional_dirs = [
            DATASET_DIR / "data" / group / subject_id / "ses-01" / "func",
            DATASET_DIR / "data" / subject_id / "ses-01" / "func",
            DATASET_DIR / "data" / group / subject_id / "func", 
            DATASET_DIR / "data" / subject_id / "func",
            DATASET_DIR / "BIDS" / group / subject_id / "ses-01" / "func",
            DATASET_DIR / "BIDS" / subject_id / "ses-01" / "func",
            DATASET_DIR / "BIDS" / group / subject_id / "func",
            DATASET_DIR / "BIDS" / subject_id / "func",
        ]
        potential_dirs.extend(additional_dirs)
    
    # Use glob pattern to find func directories anywhere in the project
    for func_dir in PROJECT_DIR.glob(f"**/{subject_id}/**/func"):
        potential_dirs.append(func_dir)
    for func_dir in PROJECT_DIR.glob(f"**/{subject_id}/func"):
        potential_dirs.append(func_dir)
    
    for path in potential_dirs:
        if path.exists():
            # Look for resting-state fMRI file (common naming conventions in BIDS)
            patterns = [
                f"{subject_id}_*task-rest*bold.nii.gz",
                f"{subject_id}_*bold.nii.gz",
                f"*{subject_id}*_bold.nii.gz",
                f"*_task-rest*_bold.nii.gz",
                f"*_bold.nii.gz",  # Fallback to any bold file
            ]
            
            for pattern in patterns:
                func_files = list(path.glob(pattern))
                if func_files:
                    logger.info(f"Found functional data for {subject_id} at {path}")
                    return func_files[0]
    
    # If we still haven't found data, look in the entire directory structure for this subject
    for bold_file in PROJECT_DIR.glob(f"**/{subject_id}/**/*bold.nii.gz"):
        logger.info(f"Found functional data for {subject_id} at {bold_file.parent}")
        return bold_file
    
    # Check fmri_processed directory directly
    if (OUTPUT_DIR / subject_id).exists():
        # Check if there's already a preprocessed file
        if (OUTPUT_DIR / subject_id / 'func_preprocessed.nii.gz').exists():
            logger.info(f"Found already preprocessed data for {subject_id}")
            return OUTPUT_DIR / subject_id / 'func_preprocessed.nii.gz'
    
    logger.warning(f"No functional data found for {subject_id}")
    return None

# Function to find fieldmap data
def find_fieldmap_data(subject_id, group):
    """Find fieldmap data for a given subject"""
    # Define possible paths based on BIDS structure
    potential_dirs = [
        DATASET_DIR / "data" / group / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "data" / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "data" / group / subject_id / "fmap",
        DATASET_DIR / "data" / subject_id / "fmap",
        DATASET_DIR / "BIDS" / group / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / group / subject_id / "fmap",
        DATASET_DIR / "BIDS" / subject_id / "fmap",
        # Add each path with 'sub-' prefix for BIDS compliance
        DATASET_DIR / "data" / group / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "data" / group / f"sub-{subject_id}" / "fmap",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "fmap",
        DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "fmap",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "fmap",
    ]
    
    # Use glob pattern to find fmap directories anywhere in the project
    for fmap_dir in PROJECT_DIR.glob(f"**/{subject_id}/**/fmap"):
        potential_dirs.append(fmap_dir)
    for fmap_dir in PROJECT_DIR.glob(f"**/{subject_id}/fmap"):
        potential_dirs.append(fmap_dir)
    
    for path in potential_dirs:
        if path.exists():
            # Look for fieldmap files with flexible pattern matching
            magnitude1_patterns = [f"{subject_id}_*magnitude1.nii.gz", "*_magnitude1.nii.gz"]
            magnitude2_patterns = [f"{subject_id}_*magnitude2.nii.gz", "*_magnitude2.nii.gz"]
            phasediff_patterns = [f"{subject_id}_*phasediff.nii.gz", "*_phasediff.nii.gz"]
            
            for mag1_pat in magnitude1_patterns:
                magnitude1 = list(path.glob(mag1_pat))
                if magnitude1:
                    break
                    
            for mag2_pat in magnitude2_patterns:
                magnitude2 = list(path.glob(mag2_pat))
                if magnitude2:
                    break
                    
            for phase_pat in phasediff_patterns:
                phasediff = list(path.glob(phase_pat))
                if phasediff:
                    break
            
            if magnitude1 and magnitude2 and phasediff:
                return {
                    'magnitude1': magnitude1[0],
                    'magnitude2': magnitude2[0],
                    'phasediff': phasediff[0],
                    'phasediff_json': Path(str(phasediff[0]).replace('.nii.gz', '.json'))
                }
    
    logger.warning(f"No complete fieldmap data found for {subject_id}")
    return None

# Update copy command to be cross-platform compatible
def copy_file(source, target):
    """Cross-platform file copy function"""
    try:
        import shutil
        shutil.copy2(str(source), str(target))
        return True
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False
        
# Main preprocessing function
def preprocess_subject(subject_id, group):
    """Run full preprocessing pipeline for a single subject"""
    logger.info(f"=== Starting preprocessing for {subject_id} (Group: {group}) ===")
    
    # Create subject output directory
    subject_output_dir = OUTPUT_DIR / subject_id
    subject_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find functional and fieldmap data
    func_file = find_func_data(subject_id, group)
    fieldmap_files = find_fieldmap_data(subject_id, group)
    
    if func_file is None:
        logger.error(f"Functional data not found for {subject_id}. Skipping.")
        return False
    
    # Copy original functional data to output directory
    output_func = subject_output_dir / 'func_original.nii.gz'
    copy_file(func_file, output_func)
    
    # Apply motion correction
    logger.info(f"{subject_id}: Applying motion correction")
    mcf_file = motion_correction(output_func, subject_output_dir)
    if mcf_file is None:
        return False
    
    # Apply distortion correction if fieldmap is available
    if fieldmap_files:
        logger.info(f"{subject_id}: Preparing fieldmap")
        fieldmap = prepare_fieldmap(fieldmap_files, subject_output_dir)
        
        if fieldmap:
            logger.info(f"{subject_id}: Applying distortion correction")
            dc_file = apply_distortion_correction(mcf_file, fieldmap, subject_output_dir)
            if dc_file is None:
                return False
        else:
            logger.warning(f"{subject_id}: Skipping distortion correction due to fieldmap preparation failure")
            dc_file = mcf_file
    else:
        logger.warning(f"{subject_id}: Skipping distortion correction (no fieldmap data)")
        dc_file = mcf_file
    
    # Register to MNI space
    logger.info(f"{subject_id}: Registering to MNI space")
    mni_file = register_to_mni(dc_file, subject_id, subject_output_dir)
    if mni_file is None:
        return False
    
    # Apply spatial smoothing
    logger.info(f"{subject_id}: Applying spatial smoothing")
    smooth_file = apply_smoothing(mni_file, subject_output_dir)
    if smooth_file is None:
        return False
    
    # Apply temporal filtering
    logger.info(f"{subject_id}: Applying temporal filtering")
    filtered_file = apply_temporal_filtering(smooth_file, subject_output_dir)
    if filtered_file is None:
        return False
    
    # Create symbolic link or copy final preprocessed file
    final_output = subject_output_dir / 'func_preprocessed.nii.gz'
    if os.path.exists(final_output):
        os.remove(final_output)
    
    # Use copy instead of symlink for better Windows compatibility
    copy_file(filtered_file, final_output)
    
    logger.info(f"=== Preprocessing completed for {subject_id} ===")
    return True

# Main execution
if __name__ == "__main__":
    # Get subject information
    subjects_df = load_subject_info()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Process subjects in parallel
    successful = 0
    failed = 0
    
    # Use process pool for parallel execution
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for _, row in subjects_df.iterrows():
            subject_id = row['subject_id']
            group = row['group']
            futures.append(executor.submit(preprocess_subject, subject_id, group))
        
        # Process results as they complete
        for future in futures:
            result = future.result()
            if result:
                successful += 1
            else:
                failed += 1
    
    logger.info(f"Preprocessing complete. Successful: {successful}, Failed: {failed}")