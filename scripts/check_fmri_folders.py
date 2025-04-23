#!/usr/bin/env python3

"""
check_fmri_folders.py - A simple script to check fMRI folder structure and content
"""

import os
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
FMRI_DIR = PROJECT_DIR / "fmri_processed"

print(f"Project directory: {PROJECT_DIR}")
print(f"Checking fMRI directory: {FMRI_DIR}")

# Check if directory exists
if not FMRI_DIR.exists():
    print(f"ERROR: The directory {FMRI_DIR} does not exist!")
    exit(1)

# Load subject information
try:
    subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
    print(f"Found {len(subjects_df)} subjects in all_subjects.csv")
except Exception as e:
    print(f"Error loading subjects CSV: {e}")
    subjects_df = None

# Count subjects by group
if subjects_df is not None:
    group_counts = subjects_df['group'].value_counts().to_dict()
    print("\nSubjects by group:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}")

# Check fMRI folders
print("\nChecking fMRI data folders...")
subject_folders = [f for f in os.listdir(FMRI_DIR) if os.path.isdir(os.path.join(FMRI_DIR, f))]
print(f"Found {len(subject_folders)} subject folders in fmri_processed")

# Check sample of subject folders for content
print("\nSampling 5 subject folders to check content:")
for i, folder in enumerate(subject_folders[:5]):
    folder_path = FMRI_DIR / folder
    files = list(folder_path.glob("**/*"))
    file_names = [f.name for f in files]
    nifti_files = [f for f in file_names if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    print(f"\nFolder {i+1}: {folder}")
    if not files:
        print("  No files found")
    else:
        print(f"  Total files: {len(files)}")
        print(f"  NIFTI files: {len(nifti_files)}")
        if nifti_files:
            print(f"  NIFTI examples: {', '.join(nifti_files[:3])}")

# Check if there are any NIFTI files at all
all_nifti = list(FMRI_DIR.glob("**/*.nii.gz"))
all_nifti.extend(FMRI_DIR.glob("**/*.nii"))

print(f"\nTotal NIFTI files found: {len(all_nifti)}")

# Create a summary file with details
summary_file = PROJECT_DIR / "fmri_data_summary.txt"
with open(summary_file, 'w') as f:
    f.write("fMRI Data Summary\n")
    f.write("================\n\n")
    f.write(f"Subjects in CSV: {len(subjects_df) if subjects_df is not None else 'N/A'}\n")
    f.write(f"Subject folders: {len(subject_folders)}\n")
    f.write(f"NIFTI files: {len(all_nifti)}\n\n")
    
    f.write("Subject folders:\n")
    for folder in subject_folders:
        folder_path = FMRI_DIR / folder
        nifti_count = len(list(folder_path.glob("**/*.nii.gz"))) + len(list(folder_path.glob("**/*.nii")))
        if subjects_df is not None:
            # Try to get the group for this subject
            subject_id = folder
            subject_row = subjects_df[subjects_df['subject_id'] == subject_id]
            if subject_row.empty and subject_id.startswith('sub-'):
                # Try without the 'sub-' prefix
                subject_id = subject_id[4:]
                subject_row = subjects_df[subjects_df['subject_id'] == subject_id]
                
            group = subject_row['group'].iloc[0] if not subject_row.empty else "Unknown"
        else:
            group = "Unknown"
            
        f.write(f"  {folder} (Group: {group}): {nifti_count} NIFTI files\n")
    
print(f"\nSummary written to {summary_file}")