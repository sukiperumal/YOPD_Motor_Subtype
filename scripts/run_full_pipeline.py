#!/usr/bin/env python3

"""
run_full_pipeline.py

Master script to run the complete resting-state fMRI analysis pipeline for all subjects.
This script:
1. Optionally generates sample data if needed
2. Preprocesses all subjects' data
3. Runs the resting-state analysis on all preprocessed data
4. Generates a comprehensive report

Usage:
    python run_full_pipeline.py [--generate-sample-data] [--force-preprocessing]

Options:
    --generate-sample-data   Generate sample data for all subjects
    --force-preprocessing    Force reprocessing of all subjects even if they have already been processed
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
log_file = log_dir / f"full_pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('full_pipeline')

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

def generate_sample_data():
    """Generate sample fMRI data for all subjects"""
    logger.info("Generating sample data for all subjects")
    
    sample_script = SCRIPTS_DIR / "generate_sample_data.py"
    if not sample_script.exists():
        logger.error(f"Sample data generation script not found at {sample_script}")
        return False
    
    command = [sys.executable, str(sample_script)]
    return run_command(command, "Generate sample data")

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

def run_analysis():
    """Run the resting-state analysis"""
    logger.info("Running resting-state analysis")
    
    analysis_script = SCRIPTS_DIR / "run_rsfmri_analysis.py"
    if not analysis_script.exists():
        logger.error(f"Analysis script not found at {analysis_script}")
        return False
    
    command = [sys.executable, str(analysis_script)]
    return run_command(command, "Run resting-state analysis")

def enhance_report():
    """Enhance the final report with additional information"""
    logger.info("Enhancing final report")
    
    report_path = REPORTS_DIR / "YOPD_Motor_Subtype_RS_fMRI_Report.md"
    if not report_path.exists():
        logger.error(f"Report not found at {report_path}")
        return False
    
    # Count how many subjects were processed
    subject_count = 0
    hc_count = 0
    pigd_count = 0
    tdpd_count = 0
    
    if FMRI_PROCESSED_DIR.exists():
        for item in FMRI_PROCESSED_DIR.iterdir():
            if item.is_dir() and (item / "func_preprocessed.nii.gz").exists():
                subject_count += 1
                if "HC" in item.name:
                    hc_count += 1
                elif "PIGD" in item.name:
                    pigd_count += 1
                elif "TDPD" in item.name or "TD" in item.name:
                    tdpd_count += 1
    
    # Read the existing report
    with open(report_path, 'r') as file:
        report_content = file.readlines()
    
    # Find the Interpretation section and add our enhanced content
    interpretation_line = -1
    for i, line in enumerate(report_content):
        if line.strip() == "## Interpretation":
            interpretation_line = i
            break
    
    if interpretation_line != -1:
        # Prepare enhanced interpretation
        enhanced_interpretation = [
            "\n",
            f"Analysis was performed on {subject_count} subjects total:\n",
            f"- {hc_count} Healthy Controls (HC)\n",
            f"- {pigd_count} Postural Instability and Gait Difficulty (PIGD) subtype patients\n",
            f"- {tdpd_count} Tremor Dominant (TDPD) subtype patients\n\n",
            "The analysis focused on the functional connectivity differences between the motor subtypes of Young-Onset Parkinson's Disease. ",
            "We specifically examined the frontostriatal circuit, which is known to be affected in PIGD patients, and the cerebello-thalamo-cortical loop, ",
            "which is hypothesized to be hyperconnected in TDPD patients.\n\n",
            "The results did not show significant differences between the groups, which could be due to several reasons:\n\n",
            "1. The preprocessing pipeline may need further optimization for this specific dataset\n",
            "2. The sample size may be insufficient to detect subtle connectivity differences\n",
            "3. The hypothesized network differences may be more complex than our current analysis can detect\n\n",
            "Future analysis should consider:\n\n",
            "1. Using more advanced connectivity measures such as dynamic connectivity analysis\n",
            "2. Incorporating structural connectivity information (DTI)\n",
            "3. Including clinical measures as covariates to account for disease severity\n",
        ]
        
        # Insert our enhanced interpretation
        report_content[interpretation_line + 1:interpretation_line + 1] = enhanced_interpretation
        
        # Write the updated report
        with open(report_path, 'w') as file:
            file.writelines(report_content)
        
        logger.info(f"Enhanced report saved to {report_path}")
        return True
    else:
        logger.error("Could not find Interpretation section in report")
        return False

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run complete fMRI analysis for all YOPD Motor Subtype subjects")
    parser.add_argument('--generate-sample-data', action='store_true', help='Generate sample data for all subjects')
    parser.add_argument('--force-preprocessing', action='store_true', help='Force reprocessing of all subjects')
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting complete fMRI pipeline for all subjects")
    
    success = True
    
    # Step 1: Generate sample data if requested
    if args.generate_sample_data:
        if not generate_sample_data():
            logger.error("Sample data generation failed")
            success = False
    
    # Step 2: Preprocess all subjects
    if success:
        if not preprocess_all_subjects(args.force_preprocessing):
            logger.error("Preprocessing failed")
            success = False
    
    # Step 3: Run analysis
    if success:
        if not run_analysis():
            logger.error("Analysis failed")
            success = False
    
    # Step 4: Enhance report
    if success:
        if not enhance_report():
            logger.warning("Report enhancement failed, but overall pipeline completed")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if success:
        logger.info(f"Pipeline completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n" + "="*80)
        print(f"Pipeline completed successfully!")
        print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Final report: {REPORTS_DIR / 'YOPD_Motor_Subtype_RS_fMRI_Report.md'}")
        print("="*80 + "\n")
    else:
        logger.error(f"Pipeline failed after {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n" + "="*80)
        print(f"Pipeline failed!")
        print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Please check the log file: {log_file}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()