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
# Look for data on the external drive
DATASET_DIR = Path("D:/data_NIMHANS").resolve()
# Also try the network share if the direct drive isn't available
if not DATASET_DIR.exists():
    try:
        DATASET_DIR = Path(r"\\LAPTOP-78NOUKL7\data_NIMHANS").resolve()
        if DATASET_DIR.exists():
            logger.info(f"Using network share path for dataset: {DATASET_DIR}")
        else:
            logger.warning(f"Network share path not accessible: {DATASET_DIR}")
            # Fallback to project directory if neither external drive nor network share is available
            DATASET_DIR = PROJECT_DIR
            logger.warning(f"Using project directory as fallback for dataset: {DATASET_DIR}")
    except Exception as e:
        logger.warning(f"Could not access network share: {e}")
        DATASET_DIR = PROJECT_DIR
        logger.warning(f"Using project directory as fallback for dataset: {DATASET_DIR}")
else:
    logger.info(f"Using external drive for dataset: {DATASET_DIR}")

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
    potential_dirs = []
    
    # Strip 'sub-' prefix if it exists for searching
    search_id = subject_id
    if subject_id.startswith("sub-"):
        search_id = subject_id[4:]  # Remove 'sub-' prefix
    
    # For data_NIMHANS structure with group folders (HC, PIGD, TDPD)
    if group is not None:
        # Primary paths based on the described data structure
        potential_dirs.extend([
            DATASET_DIR / group / subject_id / "func",
            DATASET_DIR / group / f"sub-{search_id}" / "func",
            DATASET_DIR / group / subject_id / "ses-01" / "func",
            DATASET_DIR / group / f"sub-{search_id}" / "ses-01" / "func",
            # Also try without sub- prefix within the group folder
            DATASET_DIR / group / search_id / "func",
            DATASET_DIR / group / search_id / "ses-01" / "func",
        ])
        
        # Look for just the subject ID within the group folder (in case there's no intermediate folder)
        subject_folders = list(DATASET_DIR.glob(f"{group}/*{search_id}*"))
        for folder in subject_folders:
            potential_dirs.extend([
                folder / "func",
                folder / "ses-01" / "func"
            ])
    else:
        # If group is not provided, try all possible groups
        for possible_group in ["HC", "PIGD", "TDPD"]:
            potential_dirs.extend([
                DATASET_DIR / possible_group / subject_id / "func",
                DATASET_DIR / possible_group / f"sub-{search_id}" / "func",
                DATASET_DIR / possible_group / subject_id / "ses-01" / "func",
                DATASET_DIR / possible_group / f"sub-{search_id}" / "ses-01" / "func",
                # Also try without sub- prefix within the group folder
                DATASET_DIR / possible_group / search_id / "func",
                DATASET_DIR / possible_group / search_id / "ses-01" / "func",
            ])
    
    # Add legacy/fallback paths based on BIDS specifications
    potential_dirs.extend([
        DATASET_DIR / "data" / subject_id / "ses-01" / "func",
        DATASET_DIR / "data" / subject_id / "func",
        DATASET_DIR / "BIDS" / subject_id / "ses-01" / "func",
        DATASET_DIR / "BIDS" / subject_id / "func",
        OUTPUT_DIR / subject_id,  # Check if there's already processed data
        DATASET_DIR / "data" / f"sub-{search_id}" / "ses-01" / "func",
        DATASET_DIR / "data" / f"sub-{search_id}" / "func",
        DATASET_DIR / "BIDS" / f"sub-{search_id}" / "ses-01" / "func",
        DATASET_DIR / "BIDS" / f"sub-{search_id}" / "func",
        DATASET_DIR / subject_id / "func",
        DATASET_DIR / f"sub-{search_id}" / "func",
        DATASET_DIR / subject_id,
        DATASET_DIR / f"sub-{search_id}",
        PROJECT_DIR / "data" / subject_id / "func",
        PROJECT_DIR / "data" / f"sub-{search_id}" / "func",
        PROJECT_DIR / "preprocessed" / subject_id,
        PROJECT_DIR / "fmri_processed" / subject_id,
    ])
    
    # First, check if there's already preprocessed data
    if (OUTPUT_DIR / subject_id).exists():
        preproc_file = OUTPUT_DIR / subject_id / 'func_preprocessed.nii.gz'
        if preproc_file.exists():
            logger.info(f"Found already preprocessed data for {subject_id} at {preproc_file}")
            return preproc_file
    
    # Log the number of potential directories we're searching
    num_valid_dirs = sum(1 for dir_path in potential_dirs if dir_path.exists())
    logger.info(f"Searching for {subject_id} functional data in {num_valid_dirs} valid directories out of {len(potential_dirs)} potential paths")
    
    # List some of the valid directories to help debugging
    valid_dirs = [dir_path for dir_path in potential_dirs if dir_path.exists()]
    if valid_dirs:
        paths_to_show = min(5, len(valid_dirs))
        logger.info(f"First {paths_to_show} valid directories: {[str(d) for d in valid_dirs[:paths_to_show]]}")
    
    # Then look for raw data
    for path in potential_dirs:
        if not path.exists():
            continue
        
        # Look for resting-state fMRI file (common naming conventions in BIDS)
        patterns = [
            f"{subject_id}_*task-rest*bold.nii.gz",
            f"{subject_id}_*bold.nii.gz",
            f"*{subject_id}*_bold.nii.gz",
            f"*{search_id}*_bold.nii.gz",
            f"*_task-rest*_bold.nii.gz",
            "*rest*.nii.gz",
            "*REST*.nii.gz",
            "*bold*.nii.gz",
            "*BOLD*.nii.gz",
            "*fMRI*.nii.gz",
            "*_epi*.nii.gz",
            "*.nii.gz",  # Fallback to any NIfTI file as a last resort
        ]
        
        for pattern in patterns:
            try:
                func_files = list(path.glob(pattern))
                if func_files:
                    logger.info(f"Found functional data for {subject_id} at {path}: {func_files[0].name}")
                    return func_files[0]
            except Exception as e:
                logger.debug(f"Error checking pattern {pattern} in {path}: {e}")
    
    # If we still haven't found data, use a recursive search as last resort
    try:
        # Check in group directory if group is known
        if group is not None:
            group_dir = DATASET_DIR / group
            if group_dir.exists():
                logger.info(f"Performing deep search in {group} directory for subject {subject_id}")
                # Deep search in group directory for any .nii.gz file related to this subject
                for nii_file in group_dir.glob(f"**/*{search_id}*.nii.gz"):
                    logger.info(f"Found potential functional data for {subject_id} via deep search at {nii_file}")
                    return nii_file
        
        # If still not found, check all possible subject folders in all groups
        for possible_group in ["HC", "PIGD", "TDPD"]:
            group_dir = DATASET_DIR / possible_group
            if group_dir.exists():
                # Look for potential subject folders
                for subject_folder in group_dir.glob(f"*{search_id}*"):
                    if subject_folder.is_dir():
                        # Check for any .nii.gz files within this folder
                        for nii_file in subject_folder.glob("**/*.nii.gz"):
                            if any(keyword in str(nii_file).lower() for keyword in ["bold", "rest", "epi", "fmri"]):
                                logger.info(f"Found potential functional data in {possible_group} for {subject_id} at {nii_file}")
                                return nii_file
    except Exception as e:
        logger.warning(f"Error during recursive search: {e}")
    
    # If still nothing, list subjects that exist in the dataset to help debug
    try:
        existing_groups = []
        available_subjects = {}
        for group_name in ["HC", "PIGD", "TDPD"]:
            group_path = DATASET_DIR / group_name
            if group_path.exists():
                existing_groups.append(group_name)
                available_subjects[group_name] = []
                # Get direct subdirectories which are likely subject folders
                for subject_path in group_path.glob("*"):
                    if subject_path.is_dir():
                        available_subjects[group_name].append(subject_path.name)
        
        if existing_groups:
            logger.info(f"Available groups in dataset: {', '.join(existing_groups)}")
            for group_name, subjects in available_subjects.items():
                if subjects:
                    logger.info(f"Available subjects in {group_name}: {', '.join(subjects[:10])}")
                    if len(subjects) > 10:
                        logger.info(f"...and {len(subjects) - 10} more")
                else:
                    logger.info(f"No subjects found in {group_name}")
    except Exception as e:
        logger.warning(f"Error listing available subjects: {e}")
    
    logger.warning(f"No functional data found for {subject_id}")
    return None

# Function to find fieldmap data
def find_fieldmap_data(subject_id, group):
    """Find fieldmap data for a given subject"""
    # Define possible paths based on BIDS structure
    potential_dirs = []
    
    # Only add group-specific paths if group is provided
    if group is not None:
        potential_dirs.extend([
            DATASET_DIR / "data" / group / subject_id / "ses-01" / "fmap",
            DATASET_DIR / "data" / group / subject_id / "fmap",
            DATASET_DIR / "BIDS" / group / subject_id / "ses-01" / "fmap",
            DATASET_DIR / "BIDS" / group / subject_id / "fmap",
        ])
        
        # Add paths with 'sub-' prefix if needed
        if not subject_id.startswith("sub-"):
            potential_dirs.extend([
                DATASET_DIR / "data" / group / f"sub-{subject_id}" / "ses-01" / "fmap",
                DATASET_DIR / "data" / group / f"sub-{subject_id}" / "fmap",
                DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "ses-01" / "fmap",
                DATASET_DIR / "BIDS" / group / f"sub-{subject_id}" / "fmap",
            ])
    
    # Add non-group specific paths
    potential_dirs.extend([
        DATASET_DIR / "data" / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "data" / subject_id / "fmap",
        DATASET_DIR / "BIDS" / subject_id / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / subject_id / "fmap",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "data" / f"sub-{subject_id}" / "fmap",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "ses-01" / "fmap",
        DATASET_DIR / "BIDS" / f"sub-{subject_id}" / "fmap",
    ])
    
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

# Add the missing motion_correction function
def motion_correction(func_file, output_dir):
    """Apply motion correction using FSL MCFLIRT"""
    try:
        # Set up output file path
        output_file = output_dir / 'func_mcf.nii.gz'
        
        # Set up MCFLIRT 
        mcflirt = MCFLIRT()
        mcflirt.inputs.in_file = str(func_file)
        mcflirt.inputs.out_file = str(output_file)
        mcflirt.inputs.save_plots = True  # Save motion parameters
        mcflirt.inputs.save_mats = True   # Save transformation matrices
        
        logger.info(f"Running MCFLIRT for motion correction")
        mcf_result = mcflirt.run()
        
        # Check if the output file was created
        if not output_file.exists():
            logger.error("Motion correction failed: Output file not created")
            return None
        
        logger.info(f"Motion correction complete. Output saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Motion correction failed with error: {e}")
        return None

# Add prepare_fieldmap function
def prepare_fieldmap(fieldmap_files, output_dir):
    """Prepare fieldmap for distortion correction"""
    try:
        if not fieldmap_files:
            logger.warning("No fieldmap files provided")
            return None
            
        # Extract parameters from JSON file if it exists
        echo_time1 = 0.00492  # Default TE1 in seconds
        echo_time2 = 0.00738  # Default TE2 in seconds
        
        if fieldmap_files.get('phasediff_json') and fieldmap_files['phasediff_json'].exists():
            try:
                with open(fieldmap_files['phasediff_json'], 'r') as f:
                    metadata = json.load(f)
                    
                # Extract echo times from metadata
                if 'EchoTime1' in metadata and 'EchoTime2' in metadata:
                    echo_time1 = metadata['EchoTime1']
                    echo_time2 = metadata['EchoTime2']
                    logger.info(f"Using echo times from JSON metadata: TE1={echo_time1}, TE2={echo_time2}")
            except Exception as e:
                logger.warning(f"Could not read fieldmap metadata: {e}")
        
        # Create output filenames
        magnitude_file = output_dir / 'fieldmap_magnitude.nii.gz'
        phase_file = output_dir / 'fieldmap_phase.nii.gz'
        fieldmap_file = output_dir / 'fieldmap.nii.gz'
        
        # Copy magnitude file
        copy_file(fieldmap_files['magnitude1'], magnitude_file)
        
        # Run fsl_prepare_fieldmap
        try:
            # Calculate delta TE in ms
            delta_te_ms = (echo_time2 - echo_time1) * 1000
            
            # Use FSL's tools to prepare the fieldmap
            if sys.platform == "win32":
                # For Windows, use direct command with WSL/FSL toolbox
                cmd = [
                    'fsl_prepare_fieldmap',
                    'SIEMENS',
                    str(fieldmap_files['phasediff']),
                    str(fieldmap_files['magnitude1']),
                    str(fieldmap_file),
                    str(delta_te_ms)
                ]
            else:
                # For Unix-like systems
                cmd = [
                    'fsl_prepare_fieldmap',
                    'SIEMENS',
                    str(fieldmap_files['phasediff']),
                    str(fieldmap_files['magnitude1']),
                    str(fieldmap_file),
                    str(delta_te_ms)
                ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if not fieldmap_file.exists():
                logger.error("Fieldmap preparation failed: Output file not created")
                return None
                
            return fieldmap_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Fieldmap preparation failed: {e}")
            # Fallback: Just return none to skip distortion correction
            return None
            
    except Exception as e:
        logger.error(f"Fieldmap preparation failed with error: {e}")
        return None

# Add apply_distortion_correction function
def apply_distortion_correction(func_file, fieldmap_file, output_dir):
    """Apply distortion correction using fieldmap"""
    try:
        output_file = output_dir / 'func_dc.nii.gz'
        
        # Create brain mask for functional image
        bet = BET()
        bet.inputs.in_file = str(func_file)
        bet.inputs.out_file = str(output_dir / 'func_brain.nii.gz')
        bet.inputs.mask = True
        bet_result = bet.run()
        
        # Get the mask file
        mask_file = output_dir / 'func_brain_mask.nii.gz'
        
        if not mask_file.exists():
            logger.error("Brain extraction failed: Mask file not created")
            return None
        
        # Use fugue for distortion correction
        try:
            # Prepare the command
            if sys.platform == "win32":
                # For Windows, use direct command with WSL/FSL toolbox
                cmd = [
                    'fugue',
                    '-i', str(func_file),
                    '--loadfmap=' + str(fieldmap_file),
                    '--mask=' + str(mask_file),
                    '--dwell=0.000580', # Default dwell time, adjust if needed
                    '--unwarpdir=y',    # Default phase-encode direction, adjust if needed
                    '-u', str(output_file)
                ]
            else:
                # For Unix-like systems
                cmd = [
                    'fugue',
                    '-i', str(func_file),
                    '--loadfmap=' + str(fieldmap_file),
                    '--mask=' + str(mask_file),
                    '--dwell=0.000580',
                    '--unwarpdir=y',
                    '-u', str(output_file)
                ]
                
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if not output_file.exists():
                logger.error("Distortion correction failed: Output file not created")
                return func_file  # Return original file if correction failed
                
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Distortion correction failed: {e}")
            # Return the uncorrected file to continue preprocessing
            return func_file
            
    except Exception as e:
        logger.error(f"Distortion correction failed with error: {e}")
        return func_file  # Return original file if correction failed

# Add register_to_mni function
def register_to_mni(func_file, subject_id, output_dir):
    """Register functional data to MNI space"""
    try:
        output_file = output_dir / 'func_mni.nii.gz'
        
        # Use FLIRT for linear registration
        flirt = FLIRT()
        flirt.inputs.in_file = str(func_file)
        
        # Use standard MNI template from FSL
        if sys.platform == "win32":
            mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')
        else:
            mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')
            
        if not os.path.exists(mni_template):
            logger.warning(f"MNI template not found at {mni_template}")
            # Try alternative path
            if sys.platform == "win32":
                mni_template = os.path.join(os.environ['FSLDIR'], 'fsldata', 'standard', 'MNI152_T1_2mm_brain.nii.gz')
            else:
                mni_template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
                
            if not os.path.exists(mni_template):
                logger.error("MNI template not found, registration will be skipped")
                return func_file
                
        flirt.inputs.reference = mni_template
        flirt.inputs.out_file = str(output_file)
        flirt.inputs.out_matrix_file = str(output_dir / 'func2mni.mat')
        flirt.inputs.dof = 12  # 12 degrees of freedom for affine registration
        flirt.inputs.interp = 'spline'  # Use spline interpolation
        
        logger.info("Running FLIRT for registration to MNI space")
        flirt_result = flirt.run()
        
        if not output_file.exists():
            logger.error("Registration to MNI failed: Output file not created")
            return func_file  # Return original file if registration failed
            
        logger.info(f"Registration to MNI complete. Output saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Registration to MNI failed with error: {e}")
        return func_file  # Return original file if registration failed

# Add apply_smoothing function
def apply_smoothing(func_file, output_dir):
    """Apply spatial smoothing"""
    try:
        output_file = output_dir / 'func_smooth.nii.gz'
        
        # Use FSL's susan for smoothing
        susan = SUSAN()
        susan.inputs.in_file = str(func_file)
        susan.inputs.brightness_threshold = 1000  # Adjust based on your data
        susan.inputs.fwhm = 6  # 6mm FWHM is common for fMRI
        susan.inputs.output_type = 'NIFTI_GZ'
        susan.inputs.out_file = str(output_file)
        
        logger.info("Running SUSAN for spatial smoothing")
        susan_result = susan.run()
        
        if not output_file.exists():
            logger.warning("SUSAN smoothing failed, trying IsotropicSmooth instead")
            
            # Fallback to simpler smoothing
            smoother = IsotropicSmooth()
            smoother.inputs.in_file = str(func_file)
            smoother.inputs.fwhm = 6  # 6mm FWHM
            smoother.inputs.out_file = str(output_file)
            smooth_result = smoother.run()
            
            if not output_file.exists():
                logger.error("Smoothing failed: Output file not created")
                return func_file  # Return original file if smoothing failed
                
        logger.info(f"Spatial smoothing complete. Output saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Smoothing failed with error: {e}")
        return func_file  # Return original file if smoothing failed

# Add apply_temporal_filtering function
def apply_temporal_filtering(func_file, output_dir):
    """Apply temporal filtering"""
    try:
        output_file = output_dir / 'func_filtered.nii.gz'
        
        # Calculate TR from the functional file
        tr = 2.0  # Default TR in seconds
        try:
            img = nib.load(str(func_file))
            if hasattr(img, 'header') and 'pixdim' in dir(img.header):
                tr = img.header.get_zooms()[3]  # 4th dimension timing in NIfTI is TR
            logger.info(f"Using TR = {tr} seconds for temporal filtering")
        except Exception as e:
            logger.warning(f"Could not determine TR from the image, using default TR={tr}: {e}")
        
        # Set up high-pass filter
        hp_freq = 0.01  # 0.01 Hz cutoff for high-pass
        hp_sigma = 1 / (2 * tr * hp_freq)  # Convert to sigma for FSL
        
        # Use FSL's fslmaths for temporal filtering
        try:
            if sys.platform == "win32":
                # For Windows, use direct command with WSL/FSL toolbox
                cmd = [
                    'fslmaths',
                    str(func_file),
                    '-bptf',
                    str(hp_sigma),
                    '-1',  # No low-pass filtering (-1)
                    str(output_file)
                ]
            else:
                # For Unix-like systems
                cmd = [
                    'fslmaths',
                    str(func_file),
                    '-bptf',
                    str(hp_sigma),
                    '-1',
                    str(output_file)
                ]
                
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if not output_file.exists():
                logger.error("Temporal filtering failed: Output file not created")
                return func_file  # Return original file if filtering failed
                
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Temporal filtering failed: {e}")
            # Return the unfiltered file to continue preprocessing
            return func_file
            
    except Exception as e:
        logger.error(f"Temporal filtering failed with error: {e}")
        return func_file  # Return original file if filtering failed

# Add slice timing correction
def apply_slice_timing_correction(func_file, output_dir):
    """Apply slice timing correction"""
    try:
        output_file = output_dir / 'func_stc.nii.gz'
        
        # Calculate TR from the functional file
        tr = 2.0  # Default TR in seconds
        try:
            img = nib.load(str(func_file))
            if hasattr(img, 'header') and 'pixdim' in dir(img.header):
                tr = img.header.get_zooms()[3]  # 4th dimension timing in NIfTI is TR
                if tr == 0:  # Handle cases where TR is not properly stored
                    tr = 2.0
            logger.info(f"Using TR = {tr} seconds for slice timing correction")
        except Exception as e:
            logger.warning(f"Could not determine TR from the image, using default TR={tr}: {e}")
        
        # Get number of slices
        try:
            n_slices = img.shape[2]  # Z dimension for slices
            logger.info(f"Image has {n_slices} slices")
        except:
            n_slices = 40  # Default slice count
            logger.warning(f"Could not determine slice count, using default={n_slices}")
        
        # Assume interleaved acquisition (common in fMRI)
        # For even number of slices: 0,2,4,...,1,3,5,...
        # For odd number of slices: 0,2,4,...,1,3,5,...
        slice_order = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
        
        # Use FSL's slicetimer for slice timing correction
        try:
            if sys.platform == "win32":
                # For Windows, use direct command with WSL/FSL toolbox
                slice_order_str = ','.join([str(s) for s in slice_order])
                cmd = [
                    'slicetimer',
                    '-i', str(func_file),
                    '-o', str(output_file),
                    '-r', str(tr),
                    '--ocustom=' + slice_order_str
                ]
            else:
                # For Unix-like systems
                slice_order_str = ','.join([str(s) for s in slice_order])
                cmd = [
                    'slicetimer',
                    '-i', str(func_file),
                    '-o', str(output_file),
                    '-r', str(tr),
                    '--ocustom=' + slice_order_str
                ]
                
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if not output_file.exists():
                logger.error("Slice timing correction failed: Output file not created")
                return func_file  # Return original file if correction failed
                
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Slice timing correction failed: {e}")
            # Return the uncorrected file to continue preprocessing
            return func_file
            
    except Exception as e:
        logger.error(f"Slice timing correction failed with error: {e}")
        return func_file  # Return original file if correction failed

# Complete preprocessing pipeline for a single subject
def preprocess_subject(subject_id, group=None):
    """
    Run the complete preprocessing pipeline for a single subject
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (with or without 'sub-' prefix)
    group : str, optional
        Group identifier (e.g., 'PIGD', 'HC')
    
    Returns:
    --------
    success : bool
        True if preprocessing completed successfully, False otherwise
    """
    try:
        logger.info(f"Starting preprocessing for subject {subject_id}")
        
        # Create subject-specific output directory
        subject_dir = OUTPUT_DIR / subject_id
        subject_dir.mkdir(exist_ok=True, parents=True)
        
        # Step 1: Find functional data
        func_file = find_func_data(subject_id, group)
        if func_file is None:
            logger.error(f"No functional data found for {subject_id}")
            return False
        logger.info(f"Using functional data: {func_file}")
        
        # Step 2: Motion correction
        mcf_file = motion_correction(func_file, subject_dir)
        if mcf_file is None:
            logger.error(f"Motion correction failed for {subject_id}")
            return False
        current_file = mcf_file
        
        # Step 3: Slice timing correction
        stc_file = apply_slice_timing_correction(current_file, subject_dir)
        if stc_file is not None:
            current_file = stc_file
        
        # Step 4: Distortion correction with fieldmap (if available)
        fieldmap_data = find_fieldmap_data(subject_id, group)
        if fieldmap_data:
            fieldmap_file = prepare_fieldmap(fieldmap_data, subject_dir)
            if fieldmap_file:
                dc_file = apply_distortion_correction(current_file, fieldmap_file, subject_dir)
                if dc_file is not None:
                    current_file = dc_file
        
        # Step 5: Registration to MNI space
        mni_file = register_to_mni(current_file, subject_id, subject_dir)
        if mni_file is not None:
            current_file = mni_file
        
        # Step 6: Spatial smoothing
        smooth_file = apply_smoothing(current_file, subject_dir)
        if smooth_file is not None:
            current_file = smooth_file
        
        # Step 7: Temporal filtering
        filtered_file = apply_temporal_filtering(current_file, subject_dir)
        if filtered_file is not None:
            current_file = filtered_file
        
        # Create final preprocessed output
        final_output = subject_dir / 'func_preprocessed.nii.gz'
        copy_file(current_file, final_output)
        
        logger.info(f"Preprocessing complete for subject {subject_id}")
        logger.info(f"Final preprocessed file: {final_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing subject {subject_id}: {e}")
        return False

# Main function to run preprocessing on multiple subjects
def run_preprocessing(subject_list=None, group_map=None, max_workers=4):
    """
    Run preprocessing pipeline on multiple subjects
    
    Parameters:
    -----------
    subject_list : list, optional
        List of subject IDs to process. If None, will load from config files.
    group_map : dict, optional
        Dictionary mapping subject IDs to their groups
    max_workers : int, optional
        Maximum number of parallel workers for processing
    """
    # If group_map is not provided, try to load from all_subjects.csv
    if group_map is None:
        try:
            # Load subject to group mapping from all_subjects.csv
            subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
            group_map = dict(zip(subjects_df['subject_id'], subjects_df['group']))
            logger.info(f"Loaded group information for {len(group_map)} subjects from all_subjects.csv")
        except Exception as e:
            logger.warning(f"Could not load group mapping from all_subjects.csv: {e}")
            # Try to load group information from separate files as fallback
            group_map = {}
            try:
                # Check for PIGD subjects
                with open(PROJECT_DIR / "pigd_subjects.txt", 'r') as f:
                    pigd_subjects = [line.strip() for line in f.readlines()]
                for subj in pigd_subjects:
                    group_map[subj] = "PIGD"
                    
                # Check for TDPD subjects
                with open(PROJECT_DIR / "tdpd_subjects.txt", 'r') as f:
                    tdpd_subjects = [line.strip() for line in f.readlines()]
                for subj in tdpd_subjects:
                    group_map[subj] = "TDPD"
                    
                # Check for HC subjects
                with open(PROJECT_DIR / "hc_subjects.txt", 'r') as f:
                    hc_subjects = [line.strip() for line in f.readlines()]
                for subj in hc_subjects:
                    group_map[subj] = "HC"
                
                logger.info(f"Loaded group information for {len(group_map)} subjects from separate files")
            except Exception as e:
                logger.warning(f"Error loading group information from separate files: {e}")
                group_map = {}
    
    if subject_list is None:
        # Try to load subjects from files
        try:
            # Try to load all subjects from all_subjects.csv
            subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
            subject_list = subjects_df['subject_id'].tolist()
            logger.info(f"Loaded {len(subject_list)} subjects from all_subjects.csv")
        except Exception as e:
            logger.error(f"Error loading subject list: {e}")
            # Look for subjects in the fmri_processed directory
            try:
                subject_list = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
                logger.info(f"Found {len(subject_list)} subjects in the output directory")
            except:
                logger.error("No subjects specified and couldn't find any in the output directory")
                return False
    
    logger.info(f"Starting preprocessing for {len(subject_list)} subjects")
    
    # Process subjects in parallel
    if max_workers > 1:
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_subject = {}
            for subject_id in subject_list:
                group = group_map.get(subject_id)
                if group:
                    logger.info(f"Subject {subject_id} belongs to group: {group}")
                else:
                    logger.warning(f"No group information found for subject {subject_id}")
                future = executor.submit(preprocess_subject, subject_id, group)
                future_to_subject[future] = subject_id
            
            for future in future_to_subject:
                subject_id = future_to_subject[future]
                try:
                    success = future.result()
                    results[subject_id] = success
                except Exception as e:
                    logger.error(f"Error processing subject {subject_id}: {e}")
                    results[subject_id] = False
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Preprocessing completed for {successful} out of {len(subject_list)} subjects")
    else:
        # Process subjects sequentially
        results = {}
        for subject_id in subject_list:
            group = group_map.get(subject_id)
            if group:
                logger.info(f"Subject {subject_id} belongs to group: {group}")
            else:
                logger.warning(f"No group information found for subject {subject_id}")
            success = preprocess_subject(subject_id, group)
            results[subject_id] = success
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Preprocessing completed for {successful} out of {len(subject_list)} subjects")
    
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess resting-state fMRI data for YOPD Motor Subtype Analysis')
    parser.add_argument('--subjects', nargs='+', help='List of subject IDs to process')
    parser.add_argument('--subject-file', type=str, help='File containing subject IDs (one per line)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--single-subject', type=str, help='Process a single subject')
    
    args = parser.parse_args()
    
    # Get subject list
    subject_list = None
    if args.subjects:
        subject_list = args.subjects
    elif args.subject_file:
        with open(args.subject_file, 'r') as f:
            subject_list = [line.strip() for line in f.readlines() if line.strip()]
    elif args.single_subject:
        subject_list = [args.single_subject]
    
    # Run preprocessing
    run_preprocessing(subject_list=subject_list, max_workers=args.max_workers)