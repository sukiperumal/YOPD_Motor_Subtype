"""
Configuration settings for the YOPD Motor Subtype Analysis Pipeline
"""

import os
from pathlib import Path

# Main directories
DATA_DIR = "/mnt/data_NIMHANS"
PROJECT_DIR = "/mnt/c/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Define atlas
ATLAS_URL = "https://www.gin.cnrs.fr/wp-content/uploads/BASC/BASC064.nii.gz"
ATLAS_NAME = "BASC064"

# Create necessary directories
os.makedirs(os.path.join(RESULTS_DIR, "connectivity"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "statistics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

# Analysis parameters
TR = 2.0  # Repetition time in seconds (adjust based on your sequence)
SMOOTHING_FWHM = 6  # Smoothing kernel in mm
HIGHPASS_CUTOFF = 100  # High-pass filter cutoff in seconds

# Group definitions
GROUPS = ["HC", "PIGD", "TDPD"]
GROUP_COLORS = {
    "HC": "green",
    "PIGD": "blue",
    "TDPD": "red"
}

# Networks of interest for analysis
NETWORKS_OF_INTEREST = {
    # Key networks implicated in PD motor subtypes
    "frontostriatal": [12, 13, 14, 15, 16],  # Example ROIs for frontostriatal network
    "cerebello_thalamo_cortical": [32, 33, 34, 35, 36],  # Example ROIs for CTC network
}