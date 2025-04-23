#!/usr/bin/env python3

"""
preprocess_all_subjects.py

This script ensures preprocessing is run for all subjects in the YOPD Motor Subtype study.
It uses the existing preprocessing pipeline but makes sure all subjects are processed.

Usage:
    python preprocess_all_subjects.py [--force]

Options:
    --force    Force reprocessing of subjects even if preprocessed data exists
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Configure logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"preprocess_all_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('preprocess_all')

# Set up paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
OUTPUT_DIR = PROJECT_DIR / "fmri_processed"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

def load_subject_info():
    """Load subject information from the all_subjects.csv file"""
    try:
        subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
        logger.info(f"Loaded {len(subjects_df)} subjects from all_subjects.csv")
        return subjects_df
    except FileNotFoundError:
        try:
            # Try alternative format or location
            subjects_df = pd.read_excel(PROJECT_DIR / "age_gender.xlsx")
            logger.info(f"Loaded {len(subjects_df)} subjects from age_gender.xlsx")
            return subjects_df
        except FileNotFoundError:
            logger.error("Subject information file not found")
            raise FileNotFoundError("Neither all_subjects.csv nor age_gender.xlsx found")

def load_group_files():
    """Load subjects from group-specific text files if they exist"""
    groups = {}
    
    # Check if group files exist
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

def is_processed(subject_id):
    """Check if a subject has already been processed"""
    processed_file = OUTPUT_DIR / subject_id / 'func_preprocessed.nii.gz'
    return processed_file.exists()

def preprocess_subject(subject_id, group, force=False):
    """Run preprocessing for a single subject"""
    if not force and is_processed(subject_id):
        logger.info(f"Subject {subject_id} already processed, skipping (use --force to reprocess)")
        return True
    
    # Call the individual subject preprocessing function from preprocess_fmri.py
    # We'll import the module to avoid code duplication
    sys.path.append(str(SCRIPTS_DIR))
    try:
        from preprocess_fmri import preprocess_subject as _preprocess_subject
        return _preprocess_subject(subject_id, group)
    except ImportError:
        # If import fails, call the script directly as a subprocess
        logger.info(f"Running external preprocessing for {subject_id} (Group: {group})")
        
        # Create a temporary script that preprocesses just this subject
        temp_script = PROJECT_DIR / f"temp_preprocess_{subject_id}.py"
        with open(temp_script, 'w') as f:
            # Use raw string or replace backslashes with forward slashes for Python paths
            scripts_dir_path = str(SCRIPTS_DIR).replace('\\', '/')
            f.write(f"""#!/usr/bin/env python3
import sys
sys.path.append(r'{scripts_dir_path}')
from preprocess_fmri import preprocess_subject
result = preprocess_subject('{subject_id}', '{group}')
sys.exit(0 if result else 1)
""")
        
        try:
            result = subprocess.run([sys.executable, str(temp_script)], check=False)
            success = result.returncode == 0
            if success:
                logger.info(f"Successfully preprocessed {subject_id}")
            else:
                logger.error(f"Failed to preprocess {subject_id}")
            return success
        finally:
            # Clean up
            if temp_script.exists():
                temp_script.unlink()

def create_subject_list():
    """Create a comprehensive list of all subjects and their groups"""
    subjects = []
    
    # First try to load from all_subjects.csv
    try:
        df = load_subject_info()
        for _, row in df.iterrows():
            subjects.append({
                'subject_id': row['subject_id'],
                'group': row['group']
            })
        return subjects
    except FileNotFoundError:
        logger.warning("Could not load subjects from CSV/Excel file")
    
    # If CSV/Excel not available, try to load from group files
    group_data = load_group_files()
    if not group_data:
        logger.error("No subject information found. Cannot continue.")
        sys.exit(1)
    
    for group, subject_list in group_data.items():
        for subject_id in subject_list:
            subjects.append({
                'subject_id': subject_id,
                'group': group
            })
    
    return subjects

def main():
    """Main function to preprocess all subjects"""
    parser = argparse.ArgumentParser(description="Preprocess all subjects in the YOPD Motor Subtype study")
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if data exists')
    args = parser.parse_args()
    
    logger.info("Starting preprocessing for all subjects")
    
    # Get list of all subjects
    subjects = create_subject_list()
    logger.info(f"Found {len(subjects)} subjects to process")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Process subjects (either in parallel or sequentially)
    successful = 0
    failed = 0
    
    # Process sequentially for better error handling and logging
    for subject in subjects:
        subject_id = subject['subject_id']
        group = subject['group']
        
        logger.info(f"Processing subject {subject_id} (Group: {group})")
        if preprocess_subject(subject_id, group, args.force):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Preprocessing complete. Successful: {successful}, Failed: {failed}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"Preprocessing complete!")
    print(f"Successfully processed: {successful} subjects")
    print(f"Failed to process: {failed} subjects")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()