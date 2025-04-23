#!/usr/bin/env python3

"""
process_single_subject.py

Script to process a single subject through the fMRI pipeline.
This script allows for targeted debugging of individual subject processing.

Usage:
    python process_single_subject.py --subject SUB-ID [--stage {preprocessing,analysis,all}]

Options:
    --subject SUB-ID       Subject ID to process (e.g., sub-YLOPD100)
    --stage STAGE          Processing stage to run (preprocessing, analysis, or all)
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime

# Set up logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"single_subject_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('single_subject')

# Set up paths - Fix for Windows compatibility
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
SCRIPTS_DIR = PROJECT_DIR / "scripts"
FMRI_PROCESSED_DIR = PROJECT_DIR / "fmri_processed"

def run_command(command, description):
    """Run a command and log its output"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        logger.info(f"Command output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command output:\n{e.stdout}")
        logger.error(f"Command error:\n{e.stderr}")
        return False

def preprocess_subject(subject_id):
    """Run preprocessing for a single subject"""
    logger.info(f"Preprocessing subject: {subject_id}")
    
    # First check if preprocess_fmri.py exists
    preprocess_script = SCRIPTS_DIR / "preprocess_fmri.py"
    if not preprocess_script.exists():
        logger.error(f"Preprocessing script not found at {preprocess_script}")
        return False
    
    # Run the preprocessing script for the specific subject
    command = [sys.executable, str(preprocess_script), "--subject", subject_id]
    
    return run_command(command, f"Preprocess subject {subject_id}")

def analyze_subject(subject_id):
    """Run analysis for a single subject"""
    logger.info(f"Analyzing subject: {subject_id}")
    
    # Check if the subject has been preprocessed
    subject_dir = FMRI_PROCESSED_DIR / subject_id
    if not subject_dir.exists() or not (subject_dir / "func_preprocessed.nii.gz").exists():
        logger.error(f"Subject {subject_id} has not been preprocessed yet")
        return False
    
    # Look for the resting state analysis script
    analysis_script = SCRIPTS_DIR / "resting_state_analysis.py"
    if not analysis_script.exists():
        logger.error(f"Analysis script not found at {analysis_script}")
        return False
    
    # Run the analysis script for the specific subject
    command = [sys.executable, str(analysis_script), "--subject", subject_id]
    
    return run_command(command, f"Analyze subject {subject_id}")

def main():
    """Main function to process a single subject"""
    parser = argparse.ArgumentParser(description="Process a single subject through the fMRI pipeline")
    parser.add_argument('--subject', required=True, help='Subject ID to process (e.g., sub-YLOPD100)')
    parser.add_argument('--stage', choices=['preprocessing', 'analysis', 'all'], default='all', 
                        help='Processing stage to run (preprocessing, analysis, or all)')
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"Starting processing for subject {args.subject}")
    
    success = True
    
    # Run the requested processing stage(s)
    if args.stage == 'preprocessing' or args.stage == 'all':
        if not preprocess_subject(args.subject):
            logger.error(f"Preprocessing failed for subject {args.subject}")
            success = False
    
    if (args.stage == 'analysis' or args.stage == 'all') and success:
        if not analyze_subject(args.subject):
            logger.error(f"Analysis failed for subject {args.subject}")
            success = False
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if success:
        logger.info(f"Processing for subject {args.subject} completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n" + "="*80)
        print(f"Processing for subject {args.subject} completed successfully!")
        print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Log file: {log_file}")
        print("="*80 + "\n")
    else:
        logger.error(f"Processing for subject {args.subject} failed after {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n" + "="*80)
        print(f"Processing for subject {args.subject} failed!")
        print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Please check the log file: {log_file}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()