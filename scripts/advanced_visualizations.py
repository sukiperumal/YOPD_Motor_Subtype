#!/usr/bin/env python3

"""
advanced_visualizations.py

This script generates advanced visualizations for the YOPD Motor Subtype project:
1. Original vs preprocessed T1 image comparison with enhanced colormaps
2. Subcortical region visualization with labels
3. Advanced analysis visualizations for each subcortical region
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import nibabel as nib
from nilearn import plotting, image, datasets
from nilearn.regions import connected_regions
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import random
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from scipy.ndimage import zoom

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
RAW_DIR = PROJECT_DIR / "raw_data"  # Directory containing original T1 images if available
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
FIRST_RESULTS_DIR = PROJECT_DIR / "first_results"
NILEARN_DIR = PROJECT_DIR / "nilearn_segmentation"
STATS_DIR = PROJECT_DIR / "stats"
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations" / "advanced"

# Create advanced visualizations directory if it doesn't exist
VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Set up styles for consistent visualizations
plt.style.use('seaborn-v0_8-whitegrid')
colors = {"HC": "#2C7BB6", "PIGD": "#D7191C", "TDPD": "#FDAE61"}
color_palette = sns.color_palette([colors["HC"], colors["PIGD"], colors["TDPD"]])
sns.set_palette(color_palette)

# Read subject data
subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
print(f"Total subjects: {len(subjects_df)}")
print(f"Groups: {subjects_df['group'].value_counts().to_dict()}")

# Load subcortical volumes 
try:
    # First try to load from stats directory (already processed data)
    subcortical_df = pd.read_csv(STATS_DIR / "all_subcortical_volumes.csv")
    print(f"Loaded processed subcortical volume data from stats directory")
    has_subcortical_data = True
except (FileNotFoundError, pd.errors.EmptyDataError):
    try:
        # If not found, load from nilearn_segmentation and process it
        nilearn_subcortical_df = pd.read_csv(NILEARN_DIR / "subcortical_volumes.csv")
        
        # Define mapping from nilearn column names to our structure naming convention
        structure_mapping = {
            'Left Accumbens': 'L_Accu',
            'Right Accumbens': 'R_Accu',
            'Left Amygdala': 'L_Amyg',
            'Right Amygdala': 'R_Amyg',
            'Left Caudate': 'L_Caud',
            'Right Caudate': 'R_Caud',
            'Left Hippocampus': 'L_Hipp',
            'Right Hippocampus': 'R_Hipp',
            'Left Pallidum': 'L_Pall',
            'Right Pallidum': 'R_Pall', 
            'Left Putamen': 'L_Putamen',
            'Right Putamen': 'R_Putamen',
            'Left Thalamus': 'L_Thal',
            'Right Thalamus': 'R_Thal',
            'Brain-Stem': 'BrStem'
        }
        
        # Store reversed mapping for visualization labels
        structure_labels = {v: k for k, v in structure_mapping.items()}
        
        # Transform the data into long format
        subcortical_data = []
        for _, row in nilearn_subcortical_df.iterrows():
            subject_id = row['subject_id']
            group = row['group']
            
            # Add 'sub-' prefix if not present
            if not subject_id.startswith('sub-'):
                subject_id = f"sub-{subject_id}"
            
            for orig_name, new_name in structure_mapping.items():
                if orig_name in row:
                    volume = row[orig_name]
                    subcortical_data.append({
                        'subject_id': subject_id,
                        'group': group,
                        'structure': new_name,
                        'volume_mm3': volume
                    })
        
        # Create DataFrame in required format
        subcortical_df = pd.DataFrame(subcortical_data)
        has_subcortical_data = True
        
        print(f"Processed subcortical volume data for {len(nilearn_subcortical_df)} subjects")
        
        # Save the transformed data to stats directory for future use
        subcortical_df.to_csv(STATS_DIR / "all_subcortical_volumes.csv", index=False)
        
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        has_subcortical_data = False
        subcortical_df = None
        print(f"Error loading subcortical volume data: {e}")

# Define key brain structures
structures = {
    "subcortical": [
        "L_Thal", "R_Thal",     # Thalamus
        "L_Caud", "R_Caud",     # Caudate nucleus
        "L_Pall", "R_Pall",     # Pallidum
        "L_Putamen", "R_Putamen", # Putamen
        "L_Hipp", "R_Hipp",     # Hippocampus
        "L_Amyg", "R_Amyg",     # Amygdala
        "L_Accu", "R_Accu",     # Nucleus accumbens
        "BrStem"                # Brain stem
    ],
    "pigd_focus": ["L_Pall", "R_Pall", "BrStem"],  # Typically affected in PIGD
    "tdpd_focus": ["L_Thal", "R_Thal", "L_Caud", "R_Caud"]  # Typically affected in TDPD
}

def find_raw_t1_file(subject_id):
    """
    Find the raw T1 file for a subject in D:\data_NIMHANS directory
    or the network share \\LAPTOP-78NOUKL7\data_NIMHANS
    with group-specific subfolders.
    """
    # Extract the group from the subject ID
    group = None
    if 'HC' in subject_id:
        group = 'HC'
    elif any(pid in subject_id for pid in ['PIGD', '109', '160', '193']):
        group = 'PIGD'
    elif any(td in subject_id for td in ['TDPD', '74', '72', '65']):
        group = 'TDPD'
    
    if not group:
        # Try to get the group from the subjects dataframe if available
        try:
            subject_group = subjects_df[subjects_df['subject_id'] == subject_id]['group'].values[0]
            group = subject_group
        except (KeyError, IndexError, AttributeError):
            pass
    
    # Extract subject ID number without the prefix
    subject_num = subject_id.replace('sub-', '').replace('YLOPD', '').replace('HC', '')
    
    # Define paths to check based on the determined group
    possible_paths = []
    
    if group:
        # Define the base directories to search - both D: drive and network share
        base_dirs = [
            Path(f"D:/data_NIMHANS/{group}"),
            Path(f"//LAPTOP-78NOUKL7/data_NIMHANS/{group}")
        ]
        
        for base_dir in base_dirs:
            # Add various possible filenames based on the subject ID
            possible_paths.extend([
                base_dir / f"{subject_id}.nii.gz",
                base_dir / f"{subject_id}.nii",
                base_dir / f"YLOPD{subject_num}.nii.gz",
                base_dir / f"YLOPD{subject_num}.nii",
                base_dir / f"{subject_num}.nii.gz",  # Just the number
                base_dir / f"{subject_num}.nii",     # Just the number
                base_dir / subject_id / f"{subject_id}.nii.gz",
                base_dir / subject_id / f"{subject_id}.nii",
                base_dir / subject_id / "anat" / f"{subject_id}_T1w.nii.gz",
                base_dir / subject_id / "anat" / f"{subject_id}_T1w.nii",
            ])
    
    # Also check the original raw data directory paths as fallback
    raw_paths = [
        RAW_DIR / subject_id / f"{subject_id}_T1w.nii.gz",
        RAW_DIR / subject_id / f"{subject_id}_T1w.nii",
        RAW_DIR / subject_id / "anat" / f"{subject_id}_T1w.nii.gz",
        RAW_DIR / subject_id / "anat" / f"{subject_id}_T1w.nii",
        RAW_DIR / subject_id / f"{subject_id}.nii.gz",
        RAW_DIR / subject_id / f"{subject_id}.nii",
        # Add more generic patterns that might exist in RAW_DIR
        RAW_DIR / f"{subject_id}.nii.gz",
        RAW_DIR / f"{subject_id}.nii",
        RAW_DIR / "anat" / f"{subject_id}.nii.gz",
        RAW_DIR / "anat" / f"{subject_id}.nii",
    ]
    possible_paths.extend(raw_paths)
    
    # Check if any of the possible paths exist
    for path in possible_paths:
        try:
            if path.exists():
                print(f"Found original T1 for {subject_id} at {path}")
                return path
        except (PermissionError, OSError):
            # Skip if there are permission issues or network errors
            continue
    
    # If no file was found but we have a preprocessed file without brain extraction, use that
    preprocessed_path = PREPROCESSED_DIR / subject_id
    if preprocessed_path.exists():
        try:
            non_brain_files = list(preprocessed_path.glob(f"{subject_id}*.nii.gz"))
            non_brain_files = [f for f in non_brain_files if "brain" not in f.name]
            if non_brain_files:
                print(f"Using preprocessed file as original for {subject_id} at {non_brain_files[0]}")
                return non_brain_files[0]
        except (PermissionError, OSError):
            pass
    
    print(f"Could not find original T1 for {subject_id} in D:\\data_NIMHANS\\{group}, network share, or other locations")
    return None

def visualize_original_vs_preprocessed():
    """
    Create visualizations comparing original T1 images to preprocessed T1 images
    with enhanced colormaps for 3 subjects from each group.
    """
    print("Generating original vs preprocessed T1 visualizations...")
    
    # Helper function to create custom color maps
    def create_custom_brain_cmap(name):
        if name == 'brain_hot':
            # A colormap that goes from black to red-yellow (hot)
            return LinearSegmentedColormap.from_list(
                'brain_hot', 
                [(0, 'black'), (0.33, '#330000'), (0.66, '#AA3300'), (1, '#FFFF00')]
            )
        elif name == 'brain_cool':
            # A blue-purple-white colormap
            return LinearSegmentedColormap.from_list(
                'brain_cool',
                [(0, 'black'), (0.33, '#000033'), (0.66, '#3333AA'), (1, '#FFFFFF')]
            )
        elif name == 'brain_modified':
            # A modified colormap showing brain tissue in red/orange and CSF in blue
            return LinearSegmentedColormap.from_list(
                'brain_modified', 
                [(0, 'black'), (0.4, '#000066'), (0.5, '#CCCCFF'), 
                 (0.6, '#993300'), (0.8, '#FF9900'), (1, '#FFFF99')]
            )
        else:
            return plt.cm.gray
    
    # Create colormaps
    raw_cmap = create_custom_brain_cmap('brain_modified')
    processed_cmap = plt.cm.nipy_spectral
    
    # Select 3 random subjects from each group
    selected_subjects = {}
    for group in ["HC", "PIGD", "TDPD"]:
        group_subjects = subjects_df[subjects_df['group'] == group]['subject_id'].tolist()
        
        # Get available preprocessed subjects
        available_subjects = []
        for subject_id in group_subjects:
            processed_file = PREPROCESSED_DIR / subject_id / f"{subject_id}_brain.nii.gz"
            if processed_file.exists():
                available_subjects.append(subject_id)
        
        if len(available_subjects) >= 3:
            # Randomly select 3 subjects
            selected = random.sample(available_subjects, 3)
            selected_subjects[group] = selected
            print(f"Selected subjects for {group}: {', '.join(selected)}")
        else:
            print(f"Warning: Not enough subjects with preprocessed data for group {group}")
            selected_subjects[group] = available_subjects
    
    # Create a figure for each group
    for group, subjects in selected_subjects.items():
        for i, subject_id in enumerate(subjects):
            # Define files
            processed_file = PREPROCESSED_DIR / subject_id / f"{subject_id}_brain.nii.gz"
            original_file = find_raw_t1_file(subject_id)
            
            if not processed_file.exists():
                print(f"Preprocessed file not found for {subject_id}, skipping...")
                continue
                
            # If original file not found, try to use the non-brain extracted file in preprocessed
            if original_file is None:
                original_files = list(PREPROCESSED_DIR.glob(f"{subject_id}/{subject_id}*.nii.gz"))
                original_files = [f for f in original_files if "brain" not in f.name]
                if original_files:
                    original_file = original_files[0]
                else:
                    print(f"No original T1 file found for {subject_id}, skipping...")
                    continue
            
            # Load images
            try:
                orig_img = nib.load(str(original_file))
                proc_img = nib.load(str(processed_file))
                
                # Create figure with 2 rows (original, processed) and 3 columns (axial, sagittal, coronal)
                fig = plt.figure(figsize=(18, 12))
                fig.suptitle(f"{group}: {subject_id} - Original vs. Preprocessed T1", fontsize=20)
                
                # Original image views
                for j, (mode, title) in enumerate([('z', 'Axial'), ('x', 'Sagittal'), ('y', 'Coronal')]):
                    ax = plt.subplot(2, 3, j + 1)
                    plotting.plot_anat(orig_img, axes=ax, display_mode=mode, cut_coords=1,
                                      title=f"Original: {title}", 
                                      draw_cross=False, cmap=raw_cmap)
                
                # Preprocessed image views
                for j, (mode, title) in enumerate([('z', 'Axial'), ('x', 'Sagittal'), ('y', 'Coronal')]):
                    ax = plt.subplot(2, 3, j + 4)
                    plotting.plot_anat(proc_img, axes=ax, display_mode=mode, cut_coords=1,
                                      title=f"Preprocessed: {title}", 
                                      draw_cross=False, cmap=processed_cmap)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the overall title
                plt.savefig(VISUALIZATIONS_DIR / f"{subject_id}_original_vs_processed.png", dpi=300)
                plt.close()
                print(f"Saved comparison for {subject_id}")
                
            except Exception as e:
                print(f"Error processing images for {subject_id}: {e}")
    
    # Create a figure showcasing one subject from each group for comparison
    try:
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle("T1 Images Across Subject Groups (Preprocessed)", fontsize=20)
        
        row = 0
        for group in ["HC", "PIGD", "TDPD"]:
            if group in selected_subjects and selected_subjects[group]:
                subject_id = selected_subjects[group][0]  # Take first subject
                processed_file = PREPROCESSED_DIR / subject_id / f"{subject_id}_brain.nii.gz"
                
                if processed_file.exists():
                    img = nib.load(str(processed_file))
                    
                    # Axial view
                    ax1 = plt.subplot(3, 3, row*3 + 1)
                    plotting.plot_anat(img, axes=ax1, display_mode='z', cut_coords=1, 
                                     title=f"{group}: {subject_id} (Axial)",
                                     draw_cross=False, cmap=processed_cmap)
                    
                    # Sagittal view
                    ax2 = plt.subplot(3, 3, row*3 + 2)
                    plotting.plot_anat(img, axes=ax2, display_mode='x', cut_coords=1, 
                                     title=f"{group}: {subject_id} (Sagittal)",
                                     draw_cross=False, cmap=processed_cmap)
                    
                    # Coronal view
                    ax3 = plt.subplot(3, 3, row*3 + 3)
                    plotting.plot_anat(img, axes=ax3, display_mode='y', cut_coords=1, 
                                     title=f"{group}: {subject_id} (Coronal)",
                                     draw_cross=False, cmap=processed_cmap)
                    
                row += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(VISUALIZATIONS_DIR / "group_comparison_enhanced.png", dpi=300)
        plt.close()
        print("Saved group comparison visualization")
        
    except Exception as e:
        print(f"Error creating group comparison: {e}")
        
    print("T1 visualization comparison completed.")

def visualize_subcortical_regions_with_labels():
    """
    Create visualization of all 15 subcortical regions with labels
    """
    print("Creating subcortical region visualization with labels...")

    # Load a template MNI152 brain for visualization
    mni_template = datasets.load_mni152_template()
    
    # Get an atlas with subcortical regions
    atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    atlas_img = atlas.maps
    atlas_labels = atlas.labels
    
    # Prepare the figure
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Subcortical Brain Regions", fontsize=24)
    
    # Define the subcortical structures of interest with corresponding Harvard-Oxford atlas indices
    # Note: The indices are approximate and may need adjustment based on your atlas version
    subcortical_regions = {
        "Thalamus Left": 3,
        "Thalamus Right": 14,
        "Caudate Left": 2,
        "Caudate Right": 13,
        "Putamen Left": 1, 
        "Putamen Right": 12,
        "Pallidum Left": 0,
        "Pallidum Right": 11,
        "Hippocampus Left": 4,
        "Hippocampus Right": 15,
        "Amygdala Left": 5,
        "Amygdala Right": 16,
        "Accumbens Left": 6,
        "Accumbens Right": 17,
        "Brain-Stem": 9
    }
    
    # Create a grid for plotting (5 rows x 3 columns)
    grid = gridspec.GridSpec(5, 3, figure=fig)
    
    # Create colormap for different regions
    cmap = plt.cm.hsv
    
    # Plot each subcortical region in its own subplot
    for i, (name, atlas_index) in enumerate(subcortical_regions.items()):
        row = i // 3
        col = i % 3
        
        # Create a binary mask for this region
        region_mask = image.math_img(f'img == {atlas_index}', img=atlas_img)
        
        # Plot the region
        ax = fig.add_subplot(grid[row, col])
        
        # Plot the brain with the region highlighted
        plotting.plot_roi(region_mask, bg_img=mni_template, axes=ax, 
                         title=name, cmap=plt.cm.autumn, 
                         alpha=0.7, draw_cross=False)
    
    # Plot a 3D glass brain with all regions for the last position
    if len(subcortical_regions) < 15:  # If there's a free spot
        ax = fig.add_subplot(grid[4, 2])
        
        # Create a combined image for glass brain visualization
        regions_img = None
        for i, atlas_index in enumerate(subcortical_regions.values()):
            region_mask = image.math_img(f'img == {atlas_index}', img=atlas_img)
            if regions_img is None:
                # First region
                regions_img = image.math_img(f'img * {i+1}', img=region_mask)
            else:
                # Add subsequent regions with different values
                regions_img = image.math_img(f'img1 + img2 * {i+1}', 
                                          img1=regions_img, 
                                          img2=region_mask)
        
        # Plot 3D glass brain
        if regions_img is not None:
            plotting.plot_glass_brain(regions_img, axes=ax, 
                                    title="All Subcortical Regions", 
                                    colorbar=True, plot_abs=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(VISUALIZATIONS_DIR / "subcortical_regions_labeled.png", dpi=300)
    plt.close()
    
    # Create a second visualization showing 3D views of all regions together
    try:
        # Create a colorful visualization with all regions
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle("3D Views of Subcortical Regions", fontsize=24)
        
        grid = gridspec.GridSpec(2, 2, figure=fig)
        
        # Create a 4D image with different values for each region
        combined_img = None
        colors = plt.cm.rainbow(np.linspace(0, 1, len(subcortical_regions)))
        
        for i, atlas_index in enumerate(subcortical_regions.values()):
            region_mask = image.math_img(f'img == {atlas_index}', img=atlas_img)
            if combined_img is None:
                # First region
                combined_img = image.math_img(f'img * {i+1}', img=region_mask)
            else:
                # Add subsequent regions with different values
                combined_img = image.math_img(f'img1 + img2 * {i+1}', 
                                           img1=combined_img, 
                                           img2=region_mask)
        
        # Plot axial, sagittal, coronal, and 3D views
        if combined_img is not None:
            views = [
                ('x', 'Sagittal View', grid[0, 0]),
                ('y', 'Coronal View', grid[0, 1]), 
                ('z', 'Axial View', grid[1, 0])
            ]
            
            for view, title, pos in views:
                ax = fig.add_subplot(pos)
                plotting.plot_roi(combined_img, bg_img=mni_template, axes=ax,
                                display_mode=view, cut_coords=5,
                                title=title, colorbar=True, cmap='rainbow')
            
            # Add 3D glass brain view
            ax = fig.add_subplot(grid[1, 1])
            plotting.plot_glass_brain(combined_img, axes=ax, 
                                    title="3D Glass Brain", 
                                    colorbar=True, plot_abs=False)
            
            # Add a legend mapping colors to structures
            # First create a figure legend with colored boxes for each structure
            handles = []
            for i, name in enumerate(subcortical_regions.keys()):
                color = colors[i]
                handles.append(mpatches.Patch(color=color, label=name))
            
            # Place legend outside the main plots
            fig.legend(handles=handles, loc='lower center', ncol=5, 
                      title="Subcortical Regions", bbox_to_anchor=(0.5, 0))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Leave space for legend at bottom
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_regions_3D.png", dpi=300)
        plt.close()
    
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
    
    print("Subcortical region visualization completed.")

def advanced_subcortical_analysis():
    """
    Create advanced analysis visualizations for subcortical regions
    """
    if not has_subcortical_data:
        print("No subcortical volume data available. Skipping advanced analysis.")
        return
        
    print("Performing advanced subcortical analysis...")
    
    # 1. Create normalized volume heatmap across all subjects and structures
    print("Creating volume heatmap...")
    try:
        # Get the data in wide format for the heatmap
        pivot_data = subcortical_df.pivot_table(
            index='subject_id', 
            columns='structure', 
            values='volume_mm3'
        ).reset_index()
        
        # Add group information
        pivot_data = pd.merge(
            pivot_data, 
            subjects_df[['subject_id', 'group']], 
            on='subject_id', 
            how='left'
        )
        
        # Set the index
        pivot_data.set_index('subject_id', inplace=True)
        
        # Sort subjects by group
        pivot_data = pivot_data.sort_values('group')
        group_col = pivot_data.pop('group')
        
        # Normalize the data for better visualization
        volume_data = pivot_data.values
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(volume_data)
        normalized_df = pd.DataFrame(
            normalized_data, 
            index=pivot_data.index, 
            columns=pivot_data.columns
        )
        
        # Create a heatmap with subjects grouped by diagnostic group
        plt.figure(figsize=(16, 20))
        
        # Create a color bar for the groups
        group_colors = [colors[g] for g in group_col]
        group_lut = dict(zip(set(group_col), [colors[g] for g in set(group_col)]))
        group_handles = [mpatches.Patch(color=c, label=l) for l, c in group_lut.items()]
        
        # Create row colors for the heatmap
        row_colors = pd.DataFrame({'Group': [colors[g] for g in group_col]}, index=pivot_data.index)
        
        # Generate the heatmap
        g = sns.clustermap(
            normalized_df, 
            method='ward', 
            metric='euclidean',
            figsize=(16, 20),
            cmap='RdBu_r',
            center=0,
            row_colors=row_colors,
            row_cluster=False,  # Don't cluster rows (subjects)
            col_cluster=True,   # Cluster columns (brain structures)
            dendrogram_ratio=(.1, .2),
            colors_ratio=0.02,
            cbar_pos=(.02, .32, .03, .2),
            yticklabels=True,
            xticklabels=True
        )
        
        # Add group legend to the clustergram
        g.fig.legend(
            handles=group_handles, 
            loc='upper left', 
            bbox_to_anchor=(0.05, 0.85), 
            title='Group'
        )
        
        # Adjust the clustergram
        g.fig.suptitle('Normalized Subcortical Volumes Across All Subjects', 
                      fontsize=24, y=0.92)
        g.ax_heatmap.set_xlabel('Brain Structure')
        g.ax_heatmap.set_ylabel('Subject')
        
        # Rotate x-axis labels for readability
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_volume_heatmap.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap: {e}")
    
    # 2. Create volumetric relationship plots (correlations between structures)
    print("Creating structure correlation plots...")
    try:
        # Calculate correlation matrix between structures
        structure_corr = pivot_data.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(structure_corr, dtype=bool))
        sns.heatmap(
            structure_corr, 
            annot=True, 
            cmap="RdBu_r",
            vmin=-1, 
            vmax=1, 
            center=0,
            mask=mask,
            square=True,
            fmt=".2f"
        )
        plt.title("Correlation Between Subcortical Structure Volumes", fontsize=16)
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_correlation_matrix.png", dpi=300)
        plt.close()
        
        # Create scatter plots for highly correlated pairs
        # Find top 5 correlations (excluding self-correlations)
        corr_values = []
        for i in range(len(structure_corr.columns)):
            for j in range(i+1, len(structure_corr.columns)):
                corr_values.append((
                    structure_corr.columns[i], 
                    structure_corr.columns[j], 
                    structure_corr.iloc[i, j]
                ))
        
        # Sort by absolute correlation value
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Plot top 5 correlations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Top Correlations Between Subcortical Structures", fontsize=24)
        
        for i, (struct1, struct2, corr) in enumerate(corr_values[:6]):
            ax = axes[i//3, i%3]
            
            for group in ['HC', 'PIGD', 'TDPD']:
                # Get data for this group
                group_data = subcortical_df[subcortical_df['group'] == group]
                struct1_data = group_data[group_data['structure'] == struct1]['volume_mm3']
                struct2_data = group_data[group_data['structure'] == struct2]['volume_mm3']
                
                # Create a DataFrame for plotting
                plot_data = pd.DataFrame({
                    struct1: struct1_data.values,
                    struct2: struct2_data.values,
                    'Group': group
                })
                
                # Plot the scatter
                sns.scatterplot(
                    data=plot_data, 
                    x=struct1, 
                    y=struct2, 
                    ax=ax, 
                    alpha=0.7, 
                    label=group,
                    color=colors[group]
                )
            
            # Add correlation line
            combined_data = pd.DataFrame({
                struct1: subcortical_df[subcortical_df['structure'] == struct1]['volume_mm3'],
                struct2: subcortical_df[subcortical_df['structure'] == struct2]['volume_mm3']
            })
            
            sns.regplot(
                data=combined_data, 
                x=struct1, 
                y=struct2, 
                ax=ax, 
                scatter=False, 
                line_kws={'color': 'black'}
            )
            
            ax.set_title(f"{struct1} vs {struct2}\nr = {corr:.2f}")
            
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_correlations.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating correlation plots: {e}")
    
    # 3. Create advanced statistical comparison visualization
    print("Creating advanced statistical comparisons...")
    try:
        # Run statistical tests
        def run_stats_test(structure):
            stats_results = {}
            structure_data = subcortical_df[subcortical_df['structure'] == structure]
            
            # Get groups
            hc_data = structure_data[structure_data['group'] == 'HC']['volume_mm3']
            pigd_data = structure_data[structure_data['group'] == 'PIGD']['volume_mm3']
            tdpd_data = structure_data[structure_data['group'] == 'TDPD']['volume_mm3']
            
            # Run tests if we have enough data
            if len(hc_data) >= 3 and len(pigd_data) >= 3:
                stat, pvalue = stats.mannwhitneyu(pigd_data, hc_data, alternative='two-sided')
                stats_results['PIGD_vs_HC_p'] = pvalue
                stats_results['PIGD_vs_HC_effect'] = (pigd_data.mean() - hc_data.mean()) / hc_data.mean() * 100
            
            if len(hc_data) >= 3 and len(tdpd_data) >= 3:
                stat, pvalue = stats.mannwhitneyu(tdpd_data, hc_data, alternative='two-sided')
                stats_results['TDPD_vs_HC_p'] = pvalue
                stats_results['TDPD_vs_HC_effect'] = (tdpd_data.mean() - hc_data.mean()) / hc_data.mean() * 100
            
            if len(pigd_data) >= 3 and len(tdpd_data) >= 3:
                stat, pvalue = stats.mannwhitneyu(pigd_data, tdpd_data, alternative='two-sided')
                stats_results['PIGD_vs_TDPD_p'] = pvalue
                stats_results['PIGD_vs_TDPD_effect'] = (pigd_data.mean() - tdpd_data.mean()) / tdpd_data.mean() * 100
            
            return {
                'structure': structure,
                'HC_mean': hc_data.mean() if len(hc_data) > 0 else np.nan,
                'HC_std': hc_data.std() if len(hc_data) > 0 else np.nan,
                'PIGD_mean': pigd_data.mean() if len(pigd_data) > 0 else np.nan,
                'PIGD_std': pigd_data.std() if len(pigd_data) > 0 else np.nan,
                'TDPD_mean': tdpd_data.mean() if len(tdpd_data) > 0 else np.nan,
                'TDPD_std': tdpd_data.std() if len(tdpd_data) > 0 else np.nan,
                **stats_results
            }
        
        # Run tests for all structures
        all_stats = []
        for structure in structures['subcortical']:
            structure_stats = run_stats_test(structure)
            all_stats.append(structure_stats)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(all_stats)
        
        # Save stats to file
        stats_df.to_csv(STATS_DIR / "subcortical_advanced_stats.csv", index=False)
        
        # Create a more informative visualization showing
        # 1. Volume differences between groups
        # 2. Statistical significance
        # 3. Effect sizes
        
        plt.figure(figsize=(16, 12))
        
        # Sort structures by effect size for PIGD vs HC
        if 'PIGD_vs_HC_effect' in stats_df.columns:
            stats_df = stats_df.sort_values('PIGD_vs_HC_effect', ascending=False)
        
        # Set up plot
        x_pos = np.arange(len(stats_df))
        width = 0.25
        
        # Create bars for each group
        bars1 = plt.bar(x_pos - width, stats_df['HC_mean'], width, 
                      yerr=stats_df['HC_std'], label='HC', color=colors['HC'], alpha=0.7)
        
        bars2 = plt.bar(x_pos, stats_df['PIGD_mean'], width, 
                      yerr=stats_df['PIGD_std'], label='PIGD', color=colors['PIGD'], alpha=0.7)
        
        bars3 = plt.bar(x_pos + width, stats_df['TDPD_mean'], width, 
                      yerr=stats_df['TDPD_std'], label='TDPD', color=colors['TDPD'], alpha=0.7)
        
        # Add significance markers for PIGD vs HC
        for i, row in enumerate(stats_df.itertuples()):
            if hasattr(row, 'PIGD_vs_HC_p') and row.PIGD_vs_HC_p < 0.05:
                max_y = max(row.HC_mean + row.HC_std, row.PIGD_mean + row.PIGD_std)
                plt.plot([x_pos[i]-width, x_pos[i]], [max_y*1.05, max_y*1.05], color='black')
                plt.text(x_pos[i] - width/2, max_y*1.07, '*', ha='center', fontsize=12)
            
            if hasattr(row, 'TDPD_vs_HC_p') and row.TDPD_vs_HC_p < 0.05:
                max_y = max(row.HC_mean + row.HC_std, row.TDPD_mean + row.TDPD_std)
                plt.plot([x_pos[i]-width, x_pos[i]+width], [max_y*1.1, max_y*1.1], color='black')
                plt.text(x_pos[i], max_y*1.12, '*', ha='center', fontsize=12)
            
            if hasattr(row, 'PIGD_vs_TDPD_p') and row.PIGD_vs_TDPD_p < 0.05:
                max_y = max(row.PIGD_mean + row.PIGD_std, row.TDPD_mean + row.TDPD_std)
                plt.plot([x_pos[i], x_pos[i]+width], [max_y*1.05, max_y*1.05], color='black')
                plt.text(x_pos[i] + width/2, max_y*1.07, '*', ha='center', fontsize=12)
        
        # Add x-ticks, labels, title etc.
        plt.xlabel('Subcortical Structure', fontsize=14)
        plt.ylabel('Volume (mm³)', fontsize=14)
        plt.title('Subcortical Volumes by Group with Statistical Significance', fontsize=16)
        
        plt.xticks(x_pos, stats_df['structure'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_volumes_with_stats.png", dpi=300)
        plt.close()
        
        # Create a effect size visualization
        plt.figure(figsize=(14, 10))
        
        # Prepare data for effect size plot
        effect_data = pd.melt(
            stats_df, 
            id_vars=['structure'],
            value_vars=['PIGD_vs_HC_effect', 'TDPD_vs_HC_effect', 'PIGD_vs_TDPD_effect'],
            var_name='comparison',
            value_name='effect'
        )
        
        # Map comparison names to better labels
        effect_data['comparison'] = effect_data['comparison'].map({
            'PIGD_vs_HC_effect': 'PIGD vs HC',
            'TDPD_vs_HC_effect': 'TDPD vs HC',
            'PIGD_vs_TDPD_effect': 'PIGD vs TDPD'
        })
        
        # Create barplot of effect sizes
        sns.barplot(
            data=effect_data, 
            x='structure', 
            y='effect', 
            hue='comparison',
            palette=['#e41a1c', '#377eb8', '#4daf4a']
        )
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Subcortical Structure', fontsize=14)
        plt.ylabel('Percent Difference (%)', fontsize=14)
        plt.title('Effect Sizes (% Difference) Between Groups', fontsize=16)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "subcortical_effect_sizes.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating advanced statistical visualizations: {e}")
    
    print("Advanced subcortical analysis completed.")

def main():
    """
    Main execution function
    """
    print("Starting advanced visualizations script...")
    print(f"Output directory: {VISUALIZATIONS_DIR}")
    
    # Create original vs preprocessed T1 visualization with enhanced colormaps
    visualize_original_vs_preprocessed()
    
    # Create subcortical region visualization with labels
    visualize_subcortical_regions_with_labels()
    
    # Create advanced subcortical analysis visualizations
    advanced_subcortical_analysis()
    
    print("All advanced visualizations completed!")
    print(f"Results saved to: {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    main()