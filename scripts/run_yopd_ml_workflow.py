#!/usr/bin/env python3

"""
YOPD Motor Subtype ML Analysis - Complete Workflow

This script serves as the main entry point for the complete YOPD Motor Subtype
analysis workflow, from data preparation to machine learning analysis.

The workflow includes:
1. Data preparation: Extract features from neuroimaging data
2. Data exploration: Analyze and visualize the prepared data
3. ML analysis: Apply machine learning models to classify PD subtypes
4. Results visualization: Generate plots and reports for findings
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from pathlib import Path

# Set up project paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype").resolve()
SCRIPTS_DIR = PROJECT_DIR / "scripts"
LOG_DIR = PROJECT_DIR / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
log_filename = f'ml_workflow_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / log_filename)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ml_workflow')

def run_data_preparation(args):
    """Run the data preparation script to extract features from neuroimaging data"""
    logger.info("Starting data preparation...")
    
    try:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "data_preparation" / "run_data_preparation.py")
        ]
        
        # Add command line arguments if needed
        if args.skip_data_prep:
            cmd.append("--skip-extraction")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"Data preparation failed with return code {process.returncode}")
            logger.error(process.stderr)
            return False
        else:
            logger.info(process.stdout)
            logger.info("Data preparation completed successfully")
            return True
    except Exception as e:
        logger.error(f"Error running data preparation: {e}")
        return False

def run_data_exploration(args):
    """Run the data exploration script to analyze the prepared data"""
    logger.info("Starting data exploration...")
    
    try:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "ml_analysis" / "pd_data_exploration.py")
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"Error executing {cmd[1]}: {process.stderr}")
            return False
        else:
            logger.info(process.stdout)
            logger.info("Completed data exploration")
            return True
    except Exception as e:
        logger.error(f"Error running data exploration: {e}")
        return False

def run_pd_classification(args):
    """Run the PD subtype classification script"""
    logger.info("Starting PD subtype classification...")
    
    try:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "ml_analysis" / "pd_subtype_classifier.py")
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"Error executing {cmd[1]}: {process}")
            logger.info(process.stdout)
            return False
        else:
            logger.info(process.stdout)
            logger.info("PD subtype classification completed successfully")
            return True
    except Exception as e:
        logger.error(f"Error running PD subtype classification: {e}")
        return False

def summarize_results():
    """Generate a summary of the analysis results"""
    logger.info("Summarizing analysis results...")
    
    ml_results_dir = PROJECT_DIR / "ml_results"
    exploration_dir = ml_results_dir / "exploration"
    
    # Check for data exploration results
    exploration_summary = exploration_dir / "exploration_summary.md"
    if exploration_summary.exists():
        logger.info(f"Data exploration results available at: {exploration_summary}")
    
    # Check for ML results
    feature_importance = ml_results_dir / "feature_importance.csv"
    descriptive_done = ml_results_dir / "descriptive_analysis_done.txt"
    
    if feature_importance.exists():
        logger.info("Machine learning results found")
    elif descriptive_done.exists():
        logger.info("Descriptive statistics results found")
    else:
        logger.warning("Machine learning results not found")
    
    return

def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run YOPD Motor Subtype ML Analysis Workflow')
    parser.add_argument('--skip-data-prep', action='store_true', help='Skip data preparation step')
    parser.add_argument('--skip-exploration', action='store_true', help='Skip data exploration step')
    args = parser.parse_args()
    
    logger.info("Starting YOPD Motor Subtype ML Analysis Workflow")
    
    # Run data preparation if not skipped
    if args.skip_data_prep:
        logger.info("Skipping data preparation step as requested")
        prep_success = True
    else:
        prep_success = run_data_preparation(args)
        
    # Run data exploration if not skipped and data prep was successful
    if args.skip_exploration:
        logger.info("Skipping data exploration step as requested")
        exploration_success = True
    elif prep_success:
        exploration_success = run_data_exploration(args)
    else:
        logger.error("Data preparation failed, skipping data exploration")
        exploration_success = False
        
    # Run PD subtype classification if previous steps were successful
    if exploration_success:
        classification_success = run_pd_classification(args)
    else:
        logger.error("Data exploration failed, skipping PD subtype classification")
        classification_success = False
        
    # Summarize results
    summarize_results()
    
    logger.info("YOPD Motor Subtype ML Analysis Workflow completed")

if __name__ == "__main__":
    main()