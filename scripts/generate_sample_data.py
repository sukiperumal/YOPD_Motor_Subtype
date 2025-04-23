#!/usr/bin/env python3

"""
generate_sample_data.py

This script generates sample fMRI data for all subjects in the YOPD Motor Subtype study.
It creates realistic-looking sample data that can be used to test the preprocessing pipeline.

Usage:
    python generate_sample_data.py [--template TEMPLATE_FILE]

Options:
    --template TEMPLATE_FILE   Path to template NIfTI file to use (default: uses a simple generated volume)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor

# Configure logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"generate_data_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('generate_data')

# Set up paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
DATA_DIR = PROJECT_DIR / "sample_data"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

def load_subject_info():
    """Load subject information from all_subjects.csv file"""
    try:
        subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
        logger.info(f"Loaded {len(subjects_df)} subjects from all_subjects.csv")
        return subjects_df
    except FileNotFoundError:
        try:
            subjects_df = pd.read_excel(PROJECT_DIR / "age_gender.xlsx")
            logger.info(f"Loaded {len(subjects_df)} subjects from age_gender.xlsx")
            return subjects_df
        except FileNotFoundError:
            logger.error("Subject information file not found")
            sys.exit(1)

def load_group_files():
    """Load subjects from group-specific text files"""
    groups = {}
    
    hc_file = PROJECT_DIR / "hc_subjects.txt"
    pigd_file = PROJECT_DIR / "pigd_subjects.txt"
    tdpd_file = PROJECT_DIR / "tdpd_subjects.txt"
    
    if hc_file.exists():
        with open(hc_file, 'r') as f:
            groups['HC'] = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(groups['HC'])} subjects from hc_subjects.txt")
    
    if pigd_file.exists():
        with open(pigd_file, 'r') as f:
            groups['PIGD'] = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(groups['PIGD'])} subjects from pigd_subjects.txt")
    
    if tdpd_file.exists():
        with open(tdpd_file, 'r') as f:
            groups['TDPD'] = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(groups['TDPD'])} subjects from tdpd_subjects.txt")
    
    return groups

def create_sample_functional_data(output_path, template_file=None, dimensions=(64, 64, 64, 100), group=None):
    """
    Create sample fMRI data
    
    Args:
        output_path: Path to save the output file
        template_file: Path to a template NIfTI file (optional)
        dimensions: Dimensions of the fMRI data if no template (x, y, z, time)
        group: Subject group (HC, PIGD, TDPD) to customize data characteristics
    """
    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if template_file and Path(template_file).exists():
        # Use template file
        try:
            img = nib.load(template_file)
            data = img.get_fdata()
            
            # If template doesn't have time dimension, add one
            if len(data.shape) == 3:
                data = data[:, :, :, np.newaxis]
                data = np.repeat(data, dimensions[3], axis=3)
            
            # Add some noise and temporal fluctuations
            base_mean = np.mean(data)
            noise = np.random.normal(0, 0.05 * base_mean, data.shape)
            
            # Add time-dependent signal changes
            time_points = data.shape[3]
            time_series = np.sin(np.linspace(0, 4*np.pi, time_points)) * 0.1 * base_mean
            
            # Group-specific modifications
            if group:
                if group == 'PIGD':
                    # Reduce signal in frontostriatal regions (simplistic simulation)
                    x_mid = data.shape[0] // 2
                    y_mid = data.shape[1] // 2
                    z_mid = data.shape[2] // 2
                    data[x_mid-10:x_mid+10, y_mid-10:y_mid+10, z_mid-5:z_mid+5, :] *= 0.85
                elif group == 'TDPD':
                    # Increase signal in cerebellothalamic regions (simplistic simulation)
                    z_cerebellum = data.shape[2] // 4
                    data[:, :, z_cerebellum-5:z_cerebellum+5, :] *= 1.15
            
            # Combine original data with noise and temporal fluctuations
            for t in range(time_points):
                data[:, :, :, t] = data[:, :, :, t] + noise[:, :, :, t] + time_series[t]
            
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            
        except Exception as e:
            logger.error(f"Error processing template file: {e}")
            # Fallback to generated data
            logger.warning("Falling back to generated data")
            return create_sample_functional_data(output_path, template_file=None, dimensions=dimensions, group=group)
    else:
        # Generate synthetic data
        x, y, z, t = dimensions
        
        # Create base data
        data = np.zeros((x, y, z, t), dtype=np.float32)
        
        # Create a simulated brain structure
        # Outer shell (skull)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # Distance from center
                    dx = (i - x/2) / (x/2)
                    dy = (j - y/2) / (y/2)
                    dz = (k - z/2) / (z/2)
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Brain shape
                    if dist < 0.8:  # Brain
                        data[i, j, k, :] = 800
                    elif dist < 0.9:  # CSF
                        data[i, j, k, :] = 200
                    elif dist < 1.0:  # Skull
                        data[i, j, k, :] = 300
        
        # Add some internal structures
        # Ventricles
        vent_x = x // 2
        vent_y = y // 2
        vent_z_start = z // 3
        vent_z_end = 2 * z // 3
        data[vent_x-3:vent_x+3, vent_y-3:vent_y+3, vent_z_start:vent_z_end, :] = 100
        
        # Group-specific modifications
        if group:
            if group == 'PIGD':
                # Reduce signal in frontostriatal regions
                x_mid = x // 2
                y_mid = y // 2
                z_mid = z // 2
                data[x_mid-10:x_mid+10, y_mid-10:y_mid+10, z_mid-5:z_mid+5, :] *= 0.85
            elif group == 'TDPD':
                # Increase signal in cerebellothalamic regions
                z_cerebellum = z // 4
                data[:, :, z_cerebellum-5:z_cerebellum+5, :] *= 1.15
        
        # Add some random noise
        noise = np.random.normal(0, 20, (x, y, z, t))
        data += noise
        
        # Add temporal fluctuations (simulated BOLD response)
        for t_idx in range(t):
            # Sinusoidal fluctuation with some randomness
            fluctuation = np.sin(t_idx / 10) * 20 + np.random.normal(0, 5)
            data[:, :, :, t_idx] += fluctuation
        
        # Create affine transformation matrix (identity)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0  # 3mm voxel size
        
        # Create NIfTI image
        new_img = nib.Nifti1Image(data, affine)
    
    # Save the image
    nib.save(new_img, output_path)
    logger.info(f"Generated sample data at {output_path}")
    return True

def create_sample_fieldmap_data(output_dir, subject_id, template_mag=None, template_phase=None):
    """Create sample fieldmap data for a subject"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate magnitude images
    mag1_path = output_dir / f"{subject_id}_magnitude1.nii.gz"
    mag2_path = output_dir / f"{subject_id}_magnitude2.nii.gz"
    
    # Generate phasediff image
    phase_path = output_dir / f"{subject_id}_phasediff.nii.gz"
    
    # Generate sample data
    create_sample_functional_data(mag1_path, template_file=template_mag, dimensions=(64, 64, 64, 1))
    create_sample_functional_data(mag2_path, template_file=template_mag, dimensions=(64, 64, 64, 1))
    create_sample_functional_data(phase_path, template_file=template_phase, dimensions=(64, 64, 64, 1))
    
    # Create JSON sidecar for phasediff
    json_path = output_dir / f"{subject_id}_phasediff.json"
    with open(json_path, 'w') as f:
        json_content = {
            "EchoTime1": 0.00507,
            "EchoTime2": 0.00753,
            "IntendedFor": f"func/{subject_id}_task-rest_bold.nii.gz"
        }
        import json
        json.dump(json_content, f, indent=4)
    
    return True

def generate_data_for_subject(subject_id, group, template_func=None, template_mag=None, template_phase=None):
    """Generate all necessary sample data for a subject"""
    logger.info(f"Generating sample data for {subject_id} (Group: {group})")
    
    # Create BIDS-style directory structure
    subject_dir = DATA_DIR / f"sub-{subject_id}"
    func_dir = subject_dir / "func"
    fmap_dir = subject_dir / "fmap"
    
    func_dir.mkdir(parents=True, exist_ok=True)
    fmap_dir.mkdir(parents=True, exist_ok=True)
    
    # Create functional data
    func_path = func_dir / f"sub-{subject_id}_task-rest_bold.nii.gz"
    create_sample_functional_data(func_path, template_file=template_func, dimensions=(64, 64, 64, 100), group=group)
    
    # Create fieldmap data
    create_sample_fieldmap_data(fmap_dir, f"sub-{subject_id}", template_mag, template_phase)
    
    return True

def main():
    """Main function to generate sample data for all subjects"""
    parser = argparse.ArgumentParser(description="Generate sample fMRI data for YOPD Motor Subtype subjects")
    parser.add_argument('--template', help='Path to template NIfTI file', default=None)
    parser.add_argument('--template-mag', help='Path to template magnitude NIfTI file', default=None)
    parser.add_argument('--template-phase', help='Path to template phase NIfTI file', default=None)
    args = parser.parse_args()
    
    logger.info("Starting sample data generation")
    
    # Create sample data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of all subjects
    try:
        subjects_df = load_subject_info()
        subjects = []
        
        for _, row in subjects_df.iterrows():
            subjects.append({
                'subject_id': row['subject_id'].replace('sub-', ''),  # Remove 'sub-' prefix if present
                'group': row['group']
            })
    except Exception as e:
        logger.warning(f"Could not load subjects from CSV: {e}")
        # Try using group files as fallback
        group_data = load_group_files()
        if not group_data:
            logger.error("No subject information found. Cannot continue.")
            sys.exit(1)
        
        subjects = []
        for group, subject_list in group_data.items():
            for subject_id in subject_list:
                subjects.append({
                    'subject_id': subject_id.replace('sub-', ''),
                    'group': group
                })
    
    logger.info(f"Found {len(subjects)} subjects to generate data for")
    
    # Generate data for each subject
    successful = 0
    failed = 0
    
    for subject in subjects:
        subject_id = subject['subject_id']
        group = subject['group']
        
        try:
            if generate_data_for_subject(
                subject_id, 
                group, 
                template_func=args.template,
                template_mag=args.template_mag,
                template_phase=args.template_phase
            ):
                successful += 1
            else:
                failed += 1
                logger.error(f"Failed to generate data for {subject_id}")
        except Exception as e:
            failed += 1
            logger.error(f"Error generating data for {subject_id}: {e}")
    
    logger.info(f"Data generation complete. Successful: {successful}, Failed: {failed}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"Data generation complete!")
    print(f"Successfully generated data for {successful} subjects")
    print(f"Failed to generate data for {failed} subjects")
    print(f"Data directory: {DATA_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()