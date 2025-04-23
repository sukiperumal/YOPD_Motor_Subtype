import os
import nibabel as nib
from nilearn import datasets, image, masking
import pandas as pd
import numpy as np
import gc  # Add garbage collector
import traceback  # For detailed error reporting

# Root data path - fix path for Windows
data_dir = r"C:\Users\Pesankar\OneDrive\Documents\GitHub\YOPD_Motor_Subtype"
preprocessed_dir = os.path.join(data_dir, "preprocessed")
output_dir = os.path.join(data_dir, "nilearn_segmentation")
os.makedirs(output_dir, exist_ok=True)

print("Starting script...")
print(f"Data directory: {data_dir}")
print(f"Preprocessed directory: {preprocessed_dir}")
print(f"Output directory: {output_dir}")

# Load subject group information
def load_subject_groups():
    print("Loading subject group information...")
    groups = {}
    
    # Load HC subjects
    try:
        with open(os.path.join(data_dir, "hc_subjects.txt"), 'r') as f:
            for line in f:
                subject = line.strip()
                if subject:
                    # Extract subject ID without the "sub-" prefix
                    subject_id = subject.split("-")[-1] if subject.startswith("sub-") else subject
                    groups[subject_id] = "HC"
    except Exception as e:
        print(f"Error loading HC subjects: {e}")
    
    # Load PIGD subjects
    try:
        with open(os.path.join(data_dir, "pigd_subjects.txt"), 'r') as f:
            for line in f:
                subject = line.strip()
                if subject:
                    subject_id = subject.split("-")[-1] if subject.startswith("sub-") else subject
                    groups[subject_id] = "PIGD"
    except Exception as e:
        print(f"Error loading PIGD subjects: {e}")
    
    # Load TDPD subjects
    try:
        with open(os.path.join(data_dir, "tdpd_subjects.txt"), 'r') as f:
            for line in f:
                subject = line.strip()
                if subject:
                    subject_id = subject.split("-")[-1] if subject.startswith("sub-") else subject
                    groups[subject_id] = "TDPD"
    except Exception as e:
        print(f"Error loading TDPD subjects: {e}")
    
    print(f"Loaded {len(groups)} subject group classifications")
    return groups

# Load subject groups
subject_groups = load_subject_groups()

# Try to limit memory usage
def limit_memory():
    try:
        # Release memory
        gc.collect()
    except Exception as e:
        print(f"Error in memory management: {e}")

# Load Harvard-Oxford subcortical atlas with error handling
try:
    print("Fetching Harvard-Oxford subcortical atlas...")
    atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
    atlas_img = atlas['maps']
    labels = atlas['labels']
    print(f"Atlas loaded successfully. Shape: {atlas_img.shape}")
except Exception as e:
    print(f"Error loading atlas: {e}")
    print(traceback.format_exc())
    exit(1)

# Function to compute subcortical volumes
def extract_volumes(subject_mni_path, subject_id):
    try:
        print(f"Loading brain image from {subject_mni_path}")
        brain_img = nib.load(subject_mni_path)
        print(f"Brain image loaded. Shape: {brain_img.shape}")
        
        print("Resampling atlas to match brain image...")
        atlas_resampled = image.resample_to_img(atlas_img, brain_img, interpolation='nearest')
        print("Atlas resampled successfully")
        
        print("Getting data arrays...")
        brain_data = brain_img.get_fdata()
        atlas_data = atlas_resampled.get_fdata()
        print(f"Data loaded. Brain shape: {brain_data.shape}, Atlas shape: {atlas_data.shape}")

        voxel_volume = np.prod(brain_img.header.get_zooms())  # in mm^3
        print(f"Voxel volume: {voxel_volume} mm³")
        
        # Create a dictionary to store volumes for this subject
        volumes_dict = {}
        
        print("Processing structures...")
        for idx, label in enumerate(labels[1:], start=1):  # skip background
            print(f"Processing structure {idx}: {label}")
            structure_voxels = (atlas_data == idx).sum()
            structure_volume = structure_voxels * voxel_volume
            volumes_dict[label] = structure_volume

        # Clean up memory
        del brain_img, atlas_resampled, brain_data, atlas_data
        limit_memory()
        print(f"Successfully processed {subject_id}")
        return volumes_dict
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        print(traceback.format_exc())
        # Clean up memory even on error
        limit_memory()
        return {}

# Process subjects and store results in a new format
try:
    print("Listing subject folders...")
    all_subjects = os.listdir(preprocessed_dir)
    print(f"Found {len(all_subjects)} potential subjects")
    
    # Dictionary to store all results
    all_results = {}
    structure_set = set()  # To keep track of all structure names
    
    # Process each subject individually
    for subject_folder in all_subjects:
        subject_path = os.path.join(preprocessed_dir, subject_folder)
        
        # Skip if not a directory
        if not os.path.isdir(subject_path):
            print(f"Skipping {subject_folder} - not a directory")
            continue
            
        subject_id = subject_folder.split("-")[-1]
        mni_file = os.path.join(subject_path, f"{subject_folder}_brain_mni.nii.gz")

        if os.path.exists(mni_file):
            print(f"Extracting volumes for {subject_folder}...")
            volumes_dict = extract_volumes(mni_file, subject_id)
            
            if volumes_dict:
                # Get group if available, otherwise use "Unknown"
                group = subject_groups.get(subject_id, "Unknown")
                
                # Store results with subject_id and group
                all_results[subject_id] = {
                    "group": group,
                    **volumes_dict  # Add all structure volumes
                }
                
                # Update set of all structures
                structure_set.update(volumes_dict.keys())
                print(f"Processed and stored results for {subject_id} (Group: {group})")
            else:
                print(f"No valid results for {subject_id}")
        else:
            print(f"Warning: MNI file not found for {subject_folder}: {mni_file}")
    
    # Convert results to a DataFrame with proper format
    if all_results:
        print("Converting results to DataFrame...")
        
        # Convert to list of dictionaries for pandas
        rows = []
        for subject_id, data in all_results.items():
            row = {
                "subject_id": subject_id,
                "group": data["group"]
            }
            
            # Add structure volumes
            for structure in structure_set:
                if structure in data:
                    row[structure] = data[structure]
                else:
                    row[structure] = float('nan')  # Use NaN for missing values
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        # Ensure subject_id and group are the first columns
        cols = ['subject_id', 'group'] + [col for col in df.columns if col not in ['subject_id', 'group']]
        df = df[cols]
        
        # Save to CSV
        output_file = os.path.join(output_dir, "subcortical_volumes.csv")
        df.to_csv(output_file, index=False)
        print(f"All processing complete. Results saved to: {output_file}")
    else:
        print("No results were processed.")
    
except Exception as e:
    print(f"An error occurred: {e}")
    print(traceback.format_exc())
