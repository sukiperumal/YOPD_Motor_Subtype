#!/usr/bin/env python3

"""
run_preprocessing_only.py

Script to run only the preprocessing stage of the resting-state fMRI analysis pipeline.
This script allows for isolated execution of the preprocessing component for debugging.

Usage:
    python run_preprocessing_only.py [--force]

Options:
    --force    Force reprocessing of all subjects even if they have already been processed
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
log_file = log_dir / f"preprocessing_only_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('preprocessing_only')

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

def preprocess_all_subjects(force=False):
    """Run preprocessing for all subjects"""
    logger.info("Preprocessing all subjects")
    
    preprocess_script = SCRIPTS_DIR / "preprocess_all_subjects.py"
    if not preprocess_script.exists():
        logger.error(f"Preprocessing script not found at {preprocess_script}")
        return False
    
    command = [sys.executable, str(preprocess_script)]
    if force:
        command.append("--force")
    
    return run_command(command, "Preprocess all subjects")

def main():
    """Main function to run only the preprocessing pipeline"""
    parser = argparse.ArgumentParser(description="Run preprocessing for all YOPD Motor Subtype subjects")
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all subjects')
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting preprocessing pipeline for all subjects")
    
    # Run preprocessing
    if preprocess_all_subjects(args.force):
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("Preprocessing failed")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Preprocessing pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\n" + "="*80)
    print(f"Preprocessing complete!")
    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Log file: {log_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()