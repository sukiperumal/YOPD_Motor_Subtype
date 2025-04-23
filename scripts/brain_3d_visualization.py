#!/usr/bin/env python3

"""
brain_3d_visualization.py

This script creates advanced 3D visualizations of brain regions for the YOPD Motor Subtype project:
- Interactive 3D brain models with region-specific highlighting
- Cortical/subcortical region comparisons between groups
- Surface-based visualizations with statistical overlays
- Regional connectivity visualization

Requirements:
- nilearn
- nibabel
- matplotlib
- plotly
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from nilearn import datasets, plotting, surface
from nilearn import image as nimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import scipy.stats as stats
from matplotlib import cm

# Set paths
PROJECT_DIR = Path("c:/Users/Pesankar/OneDrive/Documents/GitHub/YOPD_Motor_Subtype")
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
STATS_DIR = PROJECT_DIR / "stats"
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"

# Create 3D visualization output directory
BRAIN_3D_DIR = VISUALIZATIONS_DIR / "brain_3d"
BRAIN_3D_DIR.mkdir(exist_ok=True, parents=True)

# Set color scheme for groups
colors = {"HC": "#2C7BB6", "PIGD": "#D7191C", "TDPD": "#FDAE61"}

# Load subject and volume data
try:
    subjects_df = pd.read_csv(PROJECT_DIR / "all_subjects.csv")
    print(f"Loaded {len(subjects_df)} subjects")
    
    # Try to load subcortical volumes
    try:
        subcortical_df = pd.read_csv(STATS_DIR / "all_subcortical_volumes.csv")
        has_subcortical = True
        print("Loaded subcortical volume data")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        has_subcortical = False
        print("Subcortical volume data not found")
except FileNotFoundError:
    subjects_df = None
    has_subcortical = False
    print("Subject data not found")

def create_whole_brain_visualization():
    """
    Create an interactive 3D visualization of the whole brain with region highlighting
    """
    print("Creating whole brain 3D visualization...")
    
    # Load MNI template
    mni_template = datasets.load_mni152_template()
    
    # Use Harvard-Oxford atlas for region definitions 
    harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    
    # Create HTML visualization using nilearn's view_img
    # This generates an interactive 3D brain visualization
    view = plotting.view_img(
        mni_template,
        bg_img=mni_template,
        opacity=0.5,
        title='Interactive 3D Brain Visualization'
    )
    
    # Save the interactive 3D visualization
    view.save_as_html(str(BRAIN_3D_DIR / "whole_brain_interactive.html"))
    print("Saved whole brain interactive 3D visualization")
    
    # Create Plotly-based 3D surface visualization (more customizable)
    try:
        # Generate a glass brain for illustrative purposes with plotly
        # Use FreeSurfer's fsaverage brain surface meshes (more detailed than default)
        fsaverage = datasets.fetch_surf_fsaverage()
        
        # Get left and right hemisphere meshes
        lh_mesh = surface.load_surf_mesh(fsaverage.pial_left)
        rh_mesh = surface.load_surf_mesh(fsaverage.pial_right)
        
        # Set up a large figure with two hemispheres side by side
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=["Left Hemisphere", "Right Hemisphere"]
        )
        
        # Add left hemisphere
        fig.add_trace(
            go.Mesh3d(
                x=lh_mesh[0][:, 0],
                y=lh_mesh[0][:, 1],
                z=lh_mesh[0][:, 2],
                i=lh_mesh[1][:, 0],
                j=lh_mesh[1][:, 1],
                k=lh_mesh[1][:, 2],
                color='gray',
                opacity=0.5,
                hoverinfo='none'
            ),
            row=1, col=1
        )
        
        # Add right hemisphere
        fig.add_trace(
            go.Mesh3d(
                x=rh_mesh[0][:, 0],
                y=rh_mesh[0][:, 1],
                z=rh_mesh[0][:, 2],
                i=rh_mesh[1][:, 0],
                j=rh_mesh[1][:, 1],
                k=rh_mesh[1][:, 2],
                color='gray',
                opacity=0.5,
                hoverinfo='none'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="3D Brain Surface Visualization",
            width=1200,
            height=800,
            scene={
                'camera': {
                    'eye': {'x': 1.5, 'y': 0, 'z': 0}
                }
            },
            scene2={
                'camera': {
                    'eye': {'x': -1.5, 'y': 0, 'z': 0}
                }
            }
        )
        
        # Save as HTML
        fig.write_html(str(BRAIN_3D_DIR / "brain_surface_3d.html"))
        
        # Also save as PNG for static viewing
        fig.write_image(str(BRAIN_3D_DIR / "brain_surface_3d.png"), scale=2)
        
    except Exception as e:
        print(f"Error creating Plotly 3D surface visualization: {e}")
    
    # Create standard brain region visualization with regions of interest
    
    # 1. Regions affected in PIGD and TDPD
    pigd_regions = ["Left Pallidum", "Right Pallidum", "Brain-Stem"]
    tdpd_regions = ["Left Thalamus", "Right Thalamus", "Left Caudate", "Right Caudate"]
    
    # 2. Create visualization for PIGD regions
    try:
        pigd_img = plotting.plot_roi(
            harvard_oxford_atlas.maps,
            bg_img=mni_template,
            colorbar=True,
            title="Brain Regions Associated with PIGD",
            output_file=str(BRAIN_3D_DIR / "pigd_regions.png")
        )
        
        # 3. Create visualization for TDPD regions
        tdpd_img = plotting.plot_roi(
            harvard_oxford_atlas.maps,
            bg_img=mni_template,
            colorbar=True,
            title="Brain Regions Associated with TDPD",
            output_file=str(BRAIN_3D_DIR / "tdpd_regions.png")
        )
    except Exception as e:
        print(f"Error creating standard brain region visualization: {e}")
    
    print("Completed whole brain 3D visualizations")

def create_region_specific_comparisons():
    """
    Create region-specific 3D visualizations for comparison between groups
    """
    if not has_subcortical:
        print("Cannot create region-specific comparisons - no subcortical volume data available")
        return
        
    print("Creating region-specific 3D comparisons...")
    
    # Get subcortical structures of interest
    subcortical_structures = [
        "L_Thal", "R_Thal",     # Thalamus
        "L_Caud", "R_Caud",     # Caudate
        "L_Pall", "R_Pall",     # Pallidum
        "L_Putamen", "R_Putamen", # Putamen
        "BrStem"                # Brainstem
    ]
    
    # Create a lookup for structure names
    structure_names = {
        "L_Thal": "Left Thalamus", "R_Thal": "Right Thalamus",
        "L_Caud": "Left Caudate", "R_Caud": "Right Caudate",
        "L_Pall": "Left Pallidum", "R_Pall": "Right Pallidum",
        "L_Putamen": "Left Putamen", "R_Putamen": "Right Putamen",
        "BrStem": "Brain Stem"
    }
    
    # Calculate mean volumes for each structure by group
    mean_volumes = subcortical_df.pivot_table(
        index='structure', 
        columns='group', 
        values='volume_mm3', 
        aggfunc='mean'
    ).reset_index()
    
    # Filter to structures of interest
    mean_volumes = mean_volumes[mean_volumes['structure'].isin(subcortical_structures)]
    
    # Create a 3D bar chart with plotly for volume comparison
    try:
        # Set up data for plotting
        structures = mean_volumes['structure'].map(structure_names)
        x_pos = np.arange(len(structures))
        
        # Create subplots (3D and 2D)
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=["3D Volume Comparison", "2D Volume Comparison"]
        )
        
        # Add 3D bars for each group
        for i, group in enumerate(['HC', 'PIGD', 'TDPD']):
            if group in mean_volumes.columns:
                fig.add_trace(
                    go.Bar3d(
                        x=[i] * len(structures),
                        y=x_pos,
                        z=[0] * len(structures),
                        dx=0.8,
                        dy=0.8,
                        dz=mean_volumes[group],
                        name=group,
                        text=structures,
                        hovertemplate='%{text}<br>Volume: %{dz:.1f} mm³',
                        colorbar=dict(title='Volume (mm³)'),
                        color=colors.get(group, 'gray')
                    ),
                    row=1, col=1
                )
        
        # Add 2D grouped bar chart
        for group in ['HC', 'PIGD', 'TDPD']:
            if group in mean_volumes.columns:
                fig.add_trace(
                    go.Bar(
                        x=structures,
                        y=mean_volumes[group],
                        name=group,
                        marker_color=colors.get(group, 'gray')
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Subcortical Volume Comparison Across Groups",
            scene=dict(
                xaxis_title="Group",
                yaxis_title="Structure",
                zaxis_title="Volume (mm³)",
                xaxis=dict(
                    tickvals=[0, 1, 2],
                    ticktext=['HC', 'PIGD', 'TDPD']
                ),
                yaxis=dict(
                    tickvals=x_pos,
                    ticktext=structures
                )
            ),
            xaxis=dict(title='Brain Structure'),
            yaxis=dict(title='Volume (mm³)'),
            barmode='group',
            width=1400,
            height=800,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )
        
        # Save as HTML
        fig.write_html(str(BRAIN_3D_DIR / "subcortical_volume_comparison_3d.html"))
        
        # Also save as PNG for static viewing
        fig.write_image(str(BRAIN_3D_DIR / "subcortical_volume_comparison_3d.png"), scale=2)
        
    except Exception as e:
        print(f"Error creating 3D volume comparison: {e}")
    
    # Create statistical comparison overlay visualization
    try:
        # Calculate statistical significance and effect sizes
        stats_data = []
        
        for struct in subcortical_structures:
            struct_data = subcortical_df[subcortical_df['structure'] == struct]
            
            # Get volumes by group
            hc_vols = struct_data[struct_data['group'] == 'HC']['volume_mm3']
            pigd_vols = struct_data[struct_data['group'] == 'PIGD']['volume_mm3']
            tdpd_vols = struct_data[struct_data['group'] == 'TDPD']['volume_mm3']
            
            # Calculate statistics if we have enough data (at least 3 subjects per group)
            if len(hc_vols) >= 3 and len(pigd_vols) >= 3:
                # PIGD vs HC
                _, pigd_hc_p = stats.ttest_ind(pigd_vols, hc_vols, equal_var=False)
                pigd_hc_effect = ((pigd_vols.mean() - hc_vols.mean()) / hc_vols.mean()) * 100
            else:
                pigd_hc_p = 1.0
                pigd_hc_effect = 0
                
            if len(hc_vols) >= 3 and len(tdpd_vols) >= 3:
                # TDPD vs HC
                _, tdpd_hc_p = stats.ttest_ind(tdpd_vols, hc_vols, equal_var=False)
                tdpd_hc_effect = ((tdpd_vols.mean() - hc_vols.mean()) / hc_vols.mean()) * 100
            else:
                tdpd_hc_p = 1.0
                tdpd_hc_effect = 0
            
            stats_data.append({
                'structure': struct,
                'structure_name': structure_names[struct],
                'pigd_hc_p': pigd_hc_p,
                'pigd_hc_effect': pigd_hc_effect,
                'tdpd_hc_p': tdpd_hc_p,
                'tdpd_hc_effect': tdpd_hc_effect,
                'hc_mean': hc_vols.mean() if len(hc_vols) > 0 else np.nan,
                'pigd_mean': pigd_vols.mean() if len(pigd_vols) > 0 else np.nan,
                'tdpd_mean': tdpd_vols.mean() if len(tdpd_vols) > 0 else np.nan
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Apply significance threshold
        alpha = 0.05
        stats_df['pigd_hc_sig'] = stats_df['pigd_hc_p'] < alpha
        stats_df['tdpd_hc_sig'] = stats_df['tdpd_hc_p'] < alpha
        
        # Create 3D glass brain with effect sizes
        # This requires a 3D volume template with region definitions
        
        # For plotting purposes, use a 2D figure with region shapes and colorbars
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot PIGD vs HC
        ax[0].set_title('PIGD vs HC Effect Size (%)')
        ax[0].set_xlabel('Structure')
        ax[0].set_ylabel('Effect Size (%)\nReduction < 0%, Increase > 0%')
        
        # Create bars with statistical significance markers
        bars = ax[0].bar(stats_df['structure_name'], stats_df['pigd_hc_effect'], 
                         color=[colors['PIGD'] if sig else 'lightgray' for sig in stats_df['pigd_hc_sig']])
        
        # Add significance markers
        for i, sig in enumerate(stats_df['pigd_hc_sig']):
            if sig:
                ax[0].text(i, stats_df['pigd_hc_effect'].iloc[i] * 1.05, '*', 
                          ha='center', va='bottom', color='black', fontsize=16)
        
        ax[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax[0].set_xticklabels(stats_df['structure_name'], rotation=45, ha='right')
        
        # Plot TDPD vs HC
        ax[1].set_title('TDPD vs HC Effect Size (%)')
        ax[1].set_xlabel('Structure')
        ax[1].set_ylabel('Effect Size (%)')
        
        bars = ax[1].bar(stats_df['structure_name'], stats_df['tdpd_hc_effect'], 
                         color=[colors['TDPD'] if sig else 'lightgray' for sig in stats_df['tdpd_hc_sig']])
        
        for i, sig in enumerate(stats_df['tdpd_hc_sig']):
            if sig:
                ax[1].text(i, stats_df['tdpd_hc_effect'].iloc[i] * 1.05, '*', 
                          ha='center', va='bottom', color='black', fontsize=16)
        
        ax[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax[1].set_xticklabels(stats_df['structure_name'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(BRAIN_3D_DIR / "subcortical_effect_sizes.png", dpi=300)
        plt.close()
        
        # Create an interactive version with plotly
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=["PIGD vs HC Effect Size (%)", "TDPD vs HC Effect Size (%)"])
        
        # Add PIGD vs HC bars
        fig.add_trace(
            go.Bar(
                x=stats_df['structure_name'], 
                y=stats_df['pigd_hc_effect'],
                text=['*' if sig else '' for sig in stats_df['pigd_hc_sig']],
                textposition='outside',
                marker_color=[colors['PIGD'] if sig else 'lightgray' for sig in stats_df['pigd_hc_sig']],
                hovertemplate='%{x}<br>Effect: %{y:.2f}%<br>p-value: %{customdata:.4f}',
                customdata=stats_df['pigd_hc_p']
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(stats_df)-0.5,
            y0=0,
            y1=0,
            line=dict(color='black', width=1),
            row=1, col=1
        )
        
        # Add TDPD vs HC bars
        fig.add_trace(
            go.Bar(
                x=stats_df['structure_name'], 
                y=stats_df['tdpd_hc_effect'],
                text=['*' if sig else '' for sig in stats_df['tdpd_hc_sig']],
                textposition='outside',
                marker_color=[colors['TDPD'] if sig else 'lightgray' for sig in stats_df['tdpd_hc_sig']],
                hovertemplate='%{x}<br>Effect: %{y:.2f}%<br>p-value: %{customdata:.4f}',
                customdata=stats_df['tdpd_hc_p']
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(stats_df)-0.5,
            y0=0,
            y1=0,
            line=dict(color='black', width=1),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Effect Sizes for Subcortical Volumes Relative to Healthy Controls",
            xaxis=dict(title='Brain Structure'),
            yaxis=dict(title='Effect Size (%)'),
            xaxis2=dict(title='Brain Structure'),
            yaxis2=dict(title='Effect Size (%)'),
            width=1400,
            height=600,
            showlegend=False
        )
        
        # Update x-axis tick rotation
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text='Effect Size<br>Reduction < 0%, Increase > 0%', row=1, col=1)
        
        # Save as HTML
        fig.write_html(str(BRAIN_3D_DIR / "subcortical_effect_sizes_interactive.html"))
        
        # Save stats table
        stats_df.to_csv(STATS_DIR / "subcortical_effect_sizes.csv", index=False)
        
    except Exception as e:
        print(f"Error creating statistical comparison overlay: {e}")
    
    print("Completed region-specific 3D comparisons")

def create_brain_connectivity_visualization():
    """
    Create a 3D visualization of brain region connectivity based on correlation
    of volumes across subjects
    """
    if not has_subcortical:
        print("Cannot create brain connectivity visualization - no subcortical volume data available")
        return
    
    print("Creating brain connectivity visualization...")
    
    try:
        # Convert to wide format for correlation analysis
        volumes_wide = subcortical_df.pivot_table(
            index='subject_id',
            columns='structure',
            values='volume_mm3'
        ).reset_index()
        
        # Calculate correlation matrix
        corr_matrix = volumes_wide.drop('subject_id', axis=1).corr()
        
        # Filter to keep only strong correlations
        threshold = 0.6
        
        # Create a 3D network graph with plotly
        nodes = {}
        edges = []
        
        # Define 3D coordinates for subcortical structures (approximate)
        coordinates = {
            "L_Thal": [-20, -20, 0],
            "R_Thal": [20, -20, 0],
            "L_Caud": [-15, 10, 0],
            "R_Caud": [15, 10, 0],
            "L_Pall": [-18, 0, -5],
            "R_Pall": [18, 0, -5],
            "L_Putamen": [-25, 5, -5],
            "R_Putamen": [25, 5, -5],
            "L_Hipp": [-25, -25, -10],
            "R_Hipp": [25, -25, -10],
            "L_Amyg": [-20, -10, -15],
            "R_Amyg": [20, -10, -15],
            "L_Accu": [-10, 10, -10],
            "R_Accu": [10, 10, -10],
            "BrStem": [0, -35, -20]
        }
        
        # Create nodes and edges data
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        
        for struct in corr_matrix.columns:
            if struct in coordinates:
                x, y, z = coordinates[struct]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_text.append(struct)
        
        # Create edges data
        edge_x = []
        edge_y = []
        edge_z = []
        edge_width = []
        edge_color = []
        
        # Create colormap for correlation values
        colorscale = px.colors.diverging.RdBu_r
        min_corr = -1
        max_corr = 1
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                struct1 = corr_matrix.columns[i]
                struct2 = corr_matrix.columns[j]
                
                if struct1 in coordinates and struct2 in coordinates:
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) >= threshold:
                        x0, y0, z0 = coordinates[struct1]
                        x1, y1, z1 = coordinates[struct2]
                        
                        # Add line coordinates with None separator
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_z.extend([z0, z1, None])
                        
                        # Map correlation to color
                        norm_corr = (corr - min_corr) / (max_corr - min_corr)
                        color_idx = min(int(norm_corr * (len(colorscale) - 1)), len(colorscale) - 1)
                        edge_color.append(colorscale[color_idx])
                        
                        # Line width based on correlation strength
                        width = 2 + 8 * abs(corr)
                        edge_width.append(width)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(
                    color=edge_color,
                    width=2
                ),
                hoverinfo='none'
            )
        )
        
        # Add nodes
        fig.add_trace(
            go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='darkblue',
                    line=dict(
                        width=2,
                        color='black'
                    )
                ),
                text=node_text,
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='3D Brain Connectivity Based on Volume Correlations',
            width=1000,
            height=800,
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False
        )
        
        # Save as HTML
        fig.write_html(str(BRAIN_3D_DIR / "brain_connectivity_3d.html"))
        
        # Also save as PNG for static viewing
        fig.write_image(str(BRAIN_3D_DIR / "brain_connectivity_3d.png"), scale=2)
        
        # Create a 2D correlation matrix heatmap for reference
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation'}
        )
        plt.title('Correlation Between Subcortical Structure Volumes')
        plt.tight_layout()
        plt.savefig(BRAIN_3D_DIR / "volume_correlation_matrix.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating brain connectivity visualization: {e}")
    
    print("Completed brain connectivity visualization")

def main():
    """
    Main function to generate all 3D brain visualizations
    """
    print("\n=== Starting 3D Brain Visualization Generator ===\n")
    
    # Create whole brain 3D visualization
    create_whole_brain_visualization()
    
    # Create region-specific 3D comparisons
    create_region_specific_comparisons()
    
    # Create brain connectivity visualization
    create_brain_connectivity_visualization()
    
    print("\n=== 3D Brain Visualization Generation Complete ===")
    print(f"Results saved to: {BRAIN_3D_DIR}")

if __name__ == "__main__":
    main()