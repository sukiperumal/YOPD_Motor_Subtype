#!/usr/bin/env python3

"""
YOPD Motor Subtype ML Analysis Runner

This script orchestrates the machine learning analysis workflow for the YOPD Motor Subtype project.
It runs the data exploration, feature analysis, and machine learning classification in sequence.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
ML_DIR = PROJECT_DIR / "ml_results"
ML_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ML_DIR / f'ml_workflow_{time.strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ml_workflow')

def run_script(script_path, description):
    """Run a Python script and log its output"""
    logger.info(f"Starting {description}...")
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log standard output and error
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"    {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"    {line}")
        
        logger.info(f"Completed {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}: {e}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"    {line}")
        
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"    {line}")
        return False

def main():
    """Main function to run the ML workflow"""
    logger.info("Starting YOPD Motor Subtype ML Analysis Workflow")
    
    # Script paths
    exploration_script = PROJECT_DIR / "scripts" / "ml_analysis" / "data_exploration.py"
    classifier_script = PROJECT_DIR / "scripts" / "ml_analysis" / "pd_subtype_classifier.py"
    
    # Step 1: Data exploration
    if run_script(exploration_script, "data exploration"):
        logger.info("Data exploration completed successfully")
    else:
        logger.error("Data exploration failed, but continuing with the workflow")
    
    # Step 2: ML classification
    if run_script(classifier_script, "PD subtype classification"):
        logger.info("PD subtype classification completed successfully")
    else:
        logger.error("PD subtype classification failed")
        
    # Check for results
    exploration_results = ML_DIR / "exploration" / "exploration_summary.md"
    ml_results = ML_DIR / "feature_importance.csv"
    
    if exploration_results.exists():
        logger.info(f"Data exploration results available at: {exploration_results}")
    else:
        logger.warning("Data exploration results not found")
    
    if ml_results.exists():
        logger.info(f"Machine learning results available at: {ml_results}")
    else:
        logger.warning("Machine learning results not found")
    
    logger.info("YOPD Motor Subtype ML Analysis Workflow completed")

if __name__ == "__main__":
    main()