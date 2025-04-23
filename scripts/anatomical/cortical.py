import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, datasets, regions, plotting
from skimage import measure, morphology
from scipy import ndimage
import time
import warnings
from pathlib import Path
# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
# Constants
CSV_PATH = "./all_subjects.csv"
PREPROCESSED_DIR = os.path.abspath("./preprocessed")
OUTPUT_DIR = os.path.abspath("./thickness_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Load atlas for parcellation
print("Loading atlas for parcellation...")
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# Store atlas map path or object and labels
atlas_img = atlas['maps']  # This could be a path or an object
atlas_labels = atlas['labels']
def estimate_cortical_thickness(brain_img):
    """
    Estimate cortical thickness from a brain image
    """
    # Load the brain image
    img = nib.load(brain_img)
    data = img.get_fdata()
    
    # Normalize image intensity to 0-1 range
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:  # Avoid division by zero
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = data
    
    # Improved tissue segmentation with adaptive thresholds
    # Calculate histogram to find better thresholds
    hist, bin_edges = np.histogram(data_norm[data_norm > 0.1], bins=100)
    
    # Use more robust thresholds
    gm_threshold_low = 0.3
    gm_threshold_high = 0.6
    wm_threshold = 0.7
    
    # Segment tissues
    gm_mask = (data_norm > gm_threshold_low) & (data_norm <= gm_threshold_high)
    wm_mask = data_norm > wm_threshold
    
    # Clean up masks with morphological operations
    gm_mask = morphology.remove_small_objects(gm_mask, min_size=50)
    gm_mask = morphology.binary_closing(gm_mask, morphology.ball(1))
    
    wm_mask = morphology.remove_small_objects(wm_mask, min_size=50)
    wm_mask = morphology.binary_closing(wm_mask, morphology.ball(1))
    
    # Calculate distance from each GM voxel to nearest WM boundary
    print("  Calculating distance maps...")
    distance_map = ndimage.distance_transform_edt(~wm_mask, sampling=img.header.get_zooms()[:3])
    
    # The thickness is twice the distance (to account for both sides of the cortex)
    thickness_map = 2 * distance_map * gm_mask
    
    # Create a new NIfTI image for the thickness map
    thickness_img = nib.Nifti1Image(thickness_map, img.affine, img.header)
    
    # Also save segmentation maps for QC
    gm_img = nib.Nifti1Image(gm_mask.astype(np.float32), img.affine, img.header)
    wm_img = nib.Nifti1Image(wm_mask.astype(np.float32), img.affine, img.header)
    
    return thickness_img, gm_img, wm_img
def register_to_mni(img, subject_output_dir, subj):
    """Register an image to MNI space for better atlas alignment"""
    from nilearn import datasets
    
    # Get MNI template
    template = datasets.load_mni152_template()
    
    # Register the image to MNI space
    print("  Registering to MNI space for better atlas alignment...")
    registered_img = image.resample_to_img(img, template, interpolation='linear')
    
    # Save the registered image
    registered_path = os.path.join(subject_output_dir, f"{subj}_thickness_mni.nii.gz")
    nib.save(registered_img, registered_path)
    
    return registered_img
def parcellate_thickness(thickness_img, subject_output_dir, subj):
    """Parcellate thickness values according to an atlas"""
    # First register the thickness map to MNI space for better atlas alignment
    mni_thickness_img = register_to_mni(thickness_img, subject_output_dir, subj)
    
    # FIXED: Check if atlas_img is a string (file path) or already a NIfTI object
    print("  Extracting regional thickness values...")
    atlas_nifti = nib.load(atlas_img) if isinstance(atlas_img, str) else atlas_img
    atlas_data = atlas_nifti.get_fdata()
    
    # Get thickness data
    thickness_data = mni_thickness_img.get_fdata()
    
    # Quick check for alignment issues
    if thickness_data.shape != atlas_data.shape:
        print(f"  ⚠️ Shape mismatch: thickness {thickness_data.shape}, atlas {atlas_data.shape}")
        # Resample atlas to match thickness dimensions
        resampled_atlas = image.resample_to_img(
            atlas_nifti, mni_thickness_img, interpolation='nearest')
        atlas_data = resampled_atlas.get_fdata()
    
    # Create a DataFrame to store regional thickness values
    results = []
    valid_regions = 0
    
    # Calculate mean thickness for each region
    for i, label in enumerate(atlas_labels):
        if i == 0:  # Skip background
            continue
            
        # Get the mask for this region
        region_mask = atlas_data == i
        
        # Skip if the region is empty
        if not np.any(region_mask):
            continue
            
        # Calculate mean thickness in this region
        region_values = thickness_data[region_mask & (thickness_data > 0)]
        
        if len(region_values) > 10:  # Only include regions with enough voxels
            region_thickness = np.mean(region_values)
            if not np.isnan(region_thickness) and region_thickness > 0:
                results.append({
                    'Subject': subj,
                    'Region': label,
                    'Mean_Thickness': region_thickness,
                    'Std_Thickness': np.std(region_values),
                    'Min_Thickness': np.min(region_values),
                    'Max_Thickness': np.max(region_values),
                    'Voxel_Count': len(region_values)
                })
                valid_regions += 1
    
    print(f"  Found {valid_regions} valid regions with thickness values")
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(os.path.join(subject_output_dir, f"{subj}_regional_thickness.csv"), index=False)
        print(f"  ✅ Saved regional thickness values to CSV")
    else:
        print(f"  ⚠️ No valid thickness results found for {subj}")
    
    return results_df
# Load CSV
df = pd.read_csv(CSV_PATH)
# Update column names if necessary
df.columns = [col.strip().lower() for col in df.columns]
# Display the first few rows to debug
print("CSV columns:", df.columns.tolist())
print("First few subject IDs from CSV:", df['subject_id'].head().tolist())
# All subjects' results
all_results = []
# Check if we actually have files in the preprocessed directory
all_dirs = [d for d in os.listdir(PREPROCESSED_DIR) if os.path.isdir(os.path.join(PREPROCESSED_DIR, d))]
print(f"Found {len(all_dirs)} subject directories in preprocessed folder: {all_dirs[:5]}")
# Process each subject
for _, row in df.iterrows():
    subject_id = row['subject_id']
    
    # Clean up any potential directory name issues
    # Check if the directory exists with or without sub- prefix
    if subject_id.startswith('sub-'):
        subj = subject_id
        alt_subj = subject_id[4:]  # without sub- prefix
    else:
        subj = f"sub-{subject_id}"
        alt_subj = subject_id
    
    print(f"\nProcessing subject: {subj}")
    
    # Try both with and without the 'sub-' prefix for flexibility
    found = False
    for test_subj in [subj, alt_subj]:
        subject_dir = os.path.join(PREPROCESSED_DIR, test_subj)
        if os.path.isdir(subject_dir):
            t1_file = os.path.join(subject_dir, f"{test_subj}_brain.nii.gz")
            
            if os.path.exists(t1_file):
                found = True
                subj = test_subj  # Use the working subject ID
                break
    
    if not found:
        # Try to find any brain file in the subject's directory
        found_alt_file = False
        for test_subj in [subj, alt_subj]:
            subject_dir = os.path.join(PREPROCESSED_DIR, test_subj)
            if os.path.isdir(subject_dir):
                nifti_files = [f for f in os.listdir(subject_dir) if f.endswith('.nii.gz')]
                brain_files = [f for f in nifti_files if 'brain' in f.lower()]
                
                if brain_files:
                    found_alt_file = True
                    t1_file = os.path.join(subject_dir, brain_files[0])
                    subj = test_subj
                    print(f"  Found alternative brain file: {brain_files[0]}")
                    break
                elif nifti_files:
                    found_alt_file = True
                    t1_file = os.path.join(subject_dir, nifti_files[0])
                    subj = test_subj
                    print(f"  No brain file found, using: {nifti_files[0]}")
                    break
        
        if not found_alt_file:
            print(f"  ⚠️ No suitable T1 file found for {subj}, skipping.")
            continue
    
    group = row.get('group', 'unknown')
    
    subject_output_dir = os.path.join(OUTPUT_DIR, subj)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    print(f"🧠 Processing cortical thickness for {subj}...")
    print(f"  Using T1 file: {t1_file}")
    start_time = time.time()
    
    try:
        # Step 1: Estimate cortical thickness
        print(f"  Estimating cortical thickness...")
        thickness_img, gm_img, wm_img = estimate_cortical_thickness(t1_file)
        
        # Save all outputs
        thickness_path = os.path.join(subject_output_dir, f"{subj}_thickness.nii.gz")
        gm_path = os.path.join(subject_output_dir, f"{subj}_gm_mask.nii.gz")
        wm_path = os.path.join(subject_output_dir, f"{subj}_wm_mask.nii.gz")
        
        nib.save(thickness_img, thickness_path)
        nib.save(gm_img, gm_path)
        nib.save(wm_img, wm_path)
        
        print(f"  ✅ Thickness map saved to {thickness_path}")
        
        # Step 2: Parcellate thickness values
        print(f"  Parcellating thickness by brain regions...")
        subject_results = parcellate_thickness(thickness_img, subject_output_dir, subj)
        
        if not subject_results.empty:
            all_results.append(subject_results)
            
            # Generate a simple visualization
            output_fig = os.path.join(subject_output_dir, f"{subj}_thickness_vis.png")
            plotting.plot_stat_map(thickness_img, threshold=0.1, 
                                  title=f"{subj} Cortical Thickness",
                                  output_file=output_fig)
            print(f"  ✅ Visualization saved to {output_fig}")
        
        elapsed = time.time() - start_time
        print(f"✅ Processing completed for {subj} in {elapsed:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error processing {subj}:\n{e}")
        import traceback
        traceback.print_exc()
# Combine all results
if all_results:
    combined_df = pd.concat(all_results)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "all_subjects_regional_thickness.csv"), index=False)
    print(f"\n✅ Combined results saved to {os.path.join(OUTPUT_DIR, 'all_subjects_regional_thickness.csv')}")
    
    # Create group-level statistics
    if 'group' in df.columns and not combined_df.empty:
        try:
            # Extract subject_id from Subject column (remove 'sub-' if present)
            combined_df['subject_id'] = combined_df['Subject'].apply(
                lambda x: x[4:] if x.startswith('sub-') else x)
            
            # Also try with 'sub-' prefix for matching
            df['match_id'] = df['subject_id'].apply(
                lambda x: x[4:] if x.startswith('sub-') else x)
            
            # Try to merge with original subject_id or match_id
            group_stats = pd.merge(
                combined_df, 
                df[['subject_id', 'match_id', 'group']], 
                left_on='subject_id',
                right_on='match_id',
                how='left'
            )
            
            if not group_stats.empty and not group_stats['group'].isna().all():
                group_summary = group_stats.groupby(['group', 'Region'])['Mean_Thickness'].agg(['mean', 'std', 'count'])
                group_summary.to_csv(os.path.join(OUTPUT_DIR, "group_thickness_stats.csv"))
                print(f"✅ Group statistics saved to {os.path.join(OUTPUT_DIR, 'group_thickness_stats.csv')}")
            else:
                print("⚠️ Could not merge subject data with group information")
        except Exception as e:
            print(f"❌ Error creating group statistics: {e}")
else:
    print("\n⚠️ No results were generated for any subjects.")
print("\n🏁 Processing complete!")
