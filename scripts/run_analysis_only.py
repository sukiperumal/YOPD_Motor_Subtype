#!/usr/bin/env python3

"""
run_analysis_only.py

Script to run only the resting-state fMRI analysis stage of the pipeline.
This script allows for isolated execution of the analysis component for debugging.

Usage:
    python run_analysis_only.py

Options:
    None
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime

# Set up logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"analysis_only_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('analysis_only')

# Set up paths - Fix for Windows compatibility
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
SCRIPTS_DIR = PROJECT_DIR / "scripts"
FMRI_PROCESSED_DIR = PROJECT_DIR / "fmri_processed"
REPORTS_DIR = PROJECT_DIR / "rsfmri_results" / "reports"

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

def verify_preprocessing():
    """Verify that preprocessing was completed"""
    logger.info("Verifying preprocessing results")
    
    if not FMRI_PROCESSED_DIR.exists():
        logger.error(f"Processed fMRI directory not found: {FMRI_PROCESSED_DIR}")
        return False
    
    # Check if at least one subject has been preprocessed
    preprocessed_subjects = [d for d in FMRI_PROCESSED_DIR.iterdir() 
                            if d.is_dir() and (d / "func_preprocessed.nii.gz").exists()]
    
    if not preprocessed_subjects:
        logger.error("No preprocessed subjects found. Please run preprocessing first.")
        return False
    
    logger.info(f"Found {len(preprocessed_subjects)} preprocessed subjects")
    return True

def run_analysis():
    """Run the resting-state analysis"""
    logger.info("Running resting-state analysis")
    
    analysis_script = SCRIPTS_DIR / "run_rsfmri_analysis.py"
    if not analysis_script.exists():
        logger.error(f"Analysis script not found at {analysis_script}")
        return False
    
    command = [sys.executable, str(analysis_script)]
    return run_command(command, "Run resting-state analysis")

def main():
    """Main function to run only the resting-state analysis pipeline"""
    start_time = time.time()
    logger.info("Starting resting-state analysis pipeline")
    
    # First verify that preprocessing has been done
    if not verify_preprocessing():
        logger.error("Preprocessing verification failed. Please run preprocessing first.")
        print("\n" + "="*80)
        print("ERROR: Preprocessing verification failed. Please run preprocessing first.")
        print(f"Log file: {log_file}")
        print("="*80 + "\n")
        return
    
    # Run analysis
    if run_analysis():
        logger.info("Resting-state analysis completed successfully")
    else:
        logger.error("Resting-state analysis failed")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Analysis pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\n" + "="*80)
    print(f"Resting-state analysis complete!")
    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Log file: {log_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()