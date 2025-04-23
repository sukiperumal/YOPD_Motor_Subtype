#!/usr/bin/env python3

"""
verify_data.py

This script scans the project directories to verify available NIFTI files
for the resting-state fMRI analysis.
"""

import os
import sys
import pandas as pd
import nibabel as nib
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('data_verification')

# Set up paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
FMRI_PROCESSED_DIR = PROJECT_DIR / "fmri_processed"
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Load subject information
def load_subject_info():
    try:
        subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
        logger.info(f"Loaded {len(subjects_df)} subjects from all_subjects.csv")
        return subjects_df
    except FileNotFoundError:
        logger.error("Subject information file not found")
        sys.exit(1)

# Check available NIFTI files
def check_available_files():
    logger.info("Checking available NIFTI files in the project directory...")
    
    # Dictionary to store results
    data_summary = {
        "fmri_processed": [],
        "preprocessed": [],
        "missing_subjects": []
    }
    
    # Get all subjects
    subjects_df = load_subject_info()
    all_subject_ids = subjects_df['subject_id'].tolist()
    
    # Check fmri_processed directory
    if FMRI_PROCESSED_DIR.exists():
        subject_dirs = list(FMRI_PROCESSED_DIR.glob("*/"))
        logger.info(f"Found {len(subject_dirs)} subject directories in fmri_processed")
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            nifti_files = list(subject_dir.glob("**/*.nii.gz"))
            
            if nifti_files:
                # Try to get additional info about the first file
                try:
                    first_file = nifti_files[0]
                    img = nib.load(str(first_file))
                    shape = img.shape
                    dimensions = f"{shape[0]}x{shape[1]}x{shape[2]}"
                    if len(shape) > 3:
                        dimensions += f" (time points: {shape[3]})"
                        
                    # Find subject group
                    subject_row = subjects_df[subjects_df['subject_id'] == subject_id]
                    if subject_row.empty:
                        # Try without 'sub-' prefix
                        if subject_id.startswith('sub-'):
                            subject_id_no_prefix = subject_id[4:]
                            subject_row = subjects_df[subjects_df['subject_id'] == subject_id_no_prefix]
                            
                    group = subject_row['group'].iloc[0] if not subject_row.empty else "Unknown"
                    
                    # Store result
                    data_summary["fmri_processed"].append({
                        "subject_id": subject_id,
                        "group": group,
                        "files_found": len(nifti_files),
                        "first_file": str(first_file.relative_to(PROJECT_DIR)),
                        "dimensions": dimensions
                    })
                except Exception as e:
                    logger.warning(f"Error reading file for {subject_id}: {e}")
                    data_summary["fmri_processed"].append({
                        "subject_id": subject_id,
                        "files_found": len(nifti_files),
                        "error": str(e)
                    })
            else:
                logger.warning(f"No NIFTI files found for {subject_id}")
                data_summary["fmri_processed"].append({
                    "subject_id": subject_id,
                    "files_found": 0
                })
    
    # Check which subjects have no data
    found_subjects = [item['subject_id'] for item in data_summary["fmri_processed"]]
    missing_subjects = []
    
    for subject_id in all_subject_ids:
        # Handle subject IDs with or without "sub-" prefix
        if subject_id not in found_subjects and f"sub-{subject_id}" not in found_subjects:
            subject_row = subjects_df[subjects_df['subject_id'] == subject_id]
            group = subject_row['group'].iloc[0] if not subject_row.empty else "Unknown"
            missing_subjects.append({"subject_id": subject_id, "group": group})
    
    data_summary["missing_subjects"] = missing_subjects
    
    # Generate report
    create_report(data_summary)
    
    return data_summary

def create_report(data_summary):
    """Create a detailed report about the data availability"""
    report_path = REPORTS_DIR / f"data_verification_report_{time.strftime('%Y%m%d')}.md"
    
    with open(report_path, 'w') as report:
        report.write("# YOPD Motor Subtype Data Verification Report\n\n")
        report.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        fmri_subjects = data_summary["fmri_processed"]
        subjects_with_data = [s for s in fmri_subjects if s.get('files_found', 0) > 0]
        report.write("## Summary\n\n")
        report.write(f"* Total subjects in all_subjects.csv: {len(load_subject_info())}\n")
        report.write(f"* Subjects with directories in fmri_processed: {len(fmri_subjects)}\n")
        report.write(f"* Subjects with at least one NIFTI file: {len(subjects_with_data)}\n")
        
        # Group statistics
        if subjects_with_data:
            groups = {}
            for subject in subjects_with_data:
                if 'group' in subject:
                    group = subject['group']
                    if group not in groups:
                        groups[group] = 0
                    groups[group] += 1
            
            report.write("\n### Group Statistics (Subjects with data)\n\n")
            for group, count in groups.items():
                report.write(f"* {group}: {count} subjects\n")
        
        # Write detailed information about files found
        report.write("\n## Found NIFTI Files\n\n")
        
        if subjects_with_data:
            report.write("| Subject ID | Group | Files Found | Dimensions | First File |\n")
            report.write("|------------|-------|-------------|------------|------------|\n")
            
            for subject in subjects_with_data:
                subject_id = subject.get('subject_id', 'Unknown')
                group = subject.get('group', 'Unknown')
                files_found = subject.get('files_found', 0)
                dimensions = subject.get('dimensions', 'N/A')
                first_file = subject.get('first_file', 'N/A')
                
                report.write(f"| {subject_id} | {group} | {files_found} | {dimensions} | {first_file} |\n")
        else:
            report.write("No subjects with NIFTI files found.\n")
        
        # Write missing subjects
        if data_summary["missing_subjects"]:
            report.write("\n## Missing Subjects\n\n")
            report.write("These subjects are in all_subjects.csv but have no NIFTI files:\n\n")
            
            report.write("| Subject ID | Group |\n")
            report.write("|------------|-------|\n")
            
            for subject in data_summary["missing_subjects"]:
                subject_id = subject.get('subject_id', 'Unknown')
                group = subject.get('group', 'Unknown')
                report.write(f"| {subject_id} | {group} |\n")
    
    logger.info(f"Report saved to {report_path}")
    return report_path

if __name__ == "__main__":
    check_available_files()