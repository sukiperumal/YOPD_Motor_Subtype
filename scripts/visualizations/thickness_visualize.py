#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cortical Thickness Visualization Tool
-------------------------------------
This script creates visualizations and statistical reports for cortical thickness 
measurements across different brain regions and subject groups (HC, PIGD, TDPD).

Key visualizations include:
1. Brain region maps with thickness distributions
2. Group comparisons across all regions
3. Individual subject thickness patterns
4. Statistical analyses and significance tests
5. Interactive dashboards for exploring the data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import nibabel as nib
from nilearn import plotting, datasets, image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import warnings
import pathlib
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set paths
BASE_DIR = os.path.abspath('.')
THICKNESS_DIR = os.path.join(BASE_DIR, 'thickness_output')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations', 'thickness')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Color schemes for consistent plotting
COLORS = {
    'HC': '#2C7BB6',  # blue
    'PIGD': '#D7191C',  # red
    'TDPD': '#FDAE61',  # orange/yellow
    'background': '#FFFFFF',
    'text': '#333333',
}

# Load data
def load_data():
    """Load all required thickness data files and prepare for visualization"""
    print("Loading data files...")
    
    # Main thickness files
    all_subjects = pd.read_csv(os.path.join(THICKNESS_DIR, 'all_subjects_regional_thickness.csv'))
    group_stats = pd.read_csv(os.path.join(THICKNESS_DIR, 'group_thickness_stats.csv'))
    
    # Extract subject ID from the Subject column (remove 'sub-' prefix if present)
    all_subjects['subject_id'] = all_subjects['Subject'].apply(
        lambda x: x[4:] if str(x).startswith('sub-') else x)
    
    # Load subject group information from all_subjects.csv
    subject_info = pd.read_csv(os.path.join(BASE_DIR, 'all_subjects.csv'))
    
    # Make column names lowercase for consistency
    subject_info.columns = [col.strip().lower() for col in subject_info.columns]
    
    # Create a clean match_id column
    subject_info['match_id'] = subject_info['subject_id'].apply(
        lambda x: x[4:] if str(x).startswith('sub-') else x)
    
    # Merge subject info with thickness data
    merged_data = pd.merge(
        all_subjects,
        subject_info[['subject_id', 'match_id', 'group']],
        left_on='subject_id',
        right_on='match_id',
        how='left'
    )
    
    print(f"Loaded data for {merged_data['Subject'].nunique()} subjects across {merged_data['Region'].nunique()} brain regions")
    
    return merged_data, group_stats, subject_info

# Visualization functions
def plot_brain_regions_thickness(data, output_path):
    """Create a brain region visualization colored by thickness values"""
    print("Generating brain region thickness map...")
    
    # Fetch atlas for visualization
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = atlas['maps']
    atlas_labels = atlas['labels']
    
    # Create a mean thickness map by region
    region_means = data.groupby('Region')['Mean_Thickness'].mean().reset_index()
    region_means = region_means.sort_values('Mean_Thickness')
    
    # Create colormap
    cmap = plt.cm.viridis
    
    # Create multiple visualizations
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # Plot 1: Sagittal view
    ax1 = plt.subplot(gs[0, 0])
    disp = plotting.plot_roi(atlas_img, cmap=cmap, figure=fig, axes=ax1,
                           title='Cortical Thickness - Sagittal View', 
                           cut_coords=(-30, 0, 30))
    ax1.set_title('Sagittal View', fontsize=14, color='black')
    
    # Plot 2: Axial view
    ax2 = plt.subplot(gs[0, 1])
    plotting.plot_roi(atlas_img, cmap=cmap, figure=fig, axes=ax2,
                    title='Cortical Thickness - Axial View', 
                    display_mode='z', cut_coords=(-10, 10, 30, 50))
    ax2.set_title('Axial View', fontsize=14, color='black')
    
    # Plot 3: 3D Surface view
    ax3 = plt.subplot(gs[1, :])
    
    # Create thickness bar chart by region
    top_regions = region_means.tail(20)  # Top 20 regions by thickness
    sns.barplot(x='Mean_Thickness', y='Region', data=top_regions, 
                palette='viridis', ax=ax3)
    ax3.set_title('Top 20 Regions by Mean Cortical Thickness', fontsize=14)
    ax3.set_xlabel('Mean Thickness', fontsize=12)
    ax3.set_ylabel('Brain Region', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'brain_regions_thickness_map.png'), dpi=300, bbox_inches='tight')
    
    # Create a glass brain visualization with region labels
    plt.figure(figsize=(15, 10))
    plotting.plot_glass_brain(atlas_img, colorbar=True, 
                            title='Cortical Regions Glass Brain View')
    plt.savefig(os.path.join(output_path, 'glass_brain_regions.png'), dpi=300, bbox_inches='tight')
    
    print("Brain region thickness map saved.")


def plot_group_comparisons(data, group_stats, output_path):
    """Create visualizations comparing cortical thickness across groups"""
    print("Generating group comparison visualizations...")
    
    # Overall thickness distribution by group
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='group', y='Mean_Thickness', data=data, palette=COLORS)
    sns.stripplot(x='group', y='Mean_Thickness', data=data, 
                 size=4, alpha=0.3, jitter=True, color='black')
    plt.title('Cortical Thickness Distribution by Group', fontsize=16)
    plt.xlabel('Group', fontsize=14)
    plt.ylabel('Mean Thickness', fontsize=14)
    plt.savefig(os.path.join(output_path, 'group_thickness_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # Heatmap of thickness by region and group
    pivot_data = group_stats.pivot(index='Region', columns='group', values='mean')
    
    # Select top regions with most difference between groups
    group_diff = pivot_data.copy()
    group_diff['max_diff'] = pivot_data.max(axis=1) - pivot_data.min(axis=1)
    top_regions_diff = group_diff.nlargest(20, 'max_diff').index
    
    # Plot heatmap for top differentiating regions
    plt.figure(figsize=(14, 12))
    sns.heatmap(pivot_data.loc[top_regions_diff], annot=True, cmap='viridis', 
               fmt='.1f', linewidths=0.5, cbar_kws={'label': 'Mean Thickness'})
    plt.title('Top 20 Regions with Greatest Group Differences in Thickness', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'group_region_thickness_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Plot bar chart for top regions
    plt.figure(figsize=(15, 10))
    top_regions_data = pivot_data.loc[top_regions_diff].reset_index()
    top_regions_long = pd.melt(top_regions_data, id_vars=['Region'], 
                              value_vars=['HC', 'PIGD', 'TDPD'],
                              var_name='Group', value_name='Mean Thickness')
    
    g = sns.catplot(x='Region', y='Mean Thickness', hue='Group', data=top_regions_long,
                   kind='bar', height=8, aspect=1.5, palette=COLORS)
    g.set_xticklabels(rotation=90)
    plt.title('Top Regions with Greatest Group Differences', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'top_regions_group_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Statistical analysis - ANOVA and post-hoc tests for each region
    significant_regions = []
    
    for region in pivot_data.index:
        region_data = data[data['Region'] == region]
        if len(region_data['group'].unique()) > 1:  # Make sure we have multiple groups
            try:
                # Run one-way ANOVA
                groups = [region_data[region_data['group'] == g]['Mean_Thickness'] for g in ['HC', 'PIGD', 'TDPD']]
                groups = [g for g in groups if len(g) > 0]  # Filter out empty groups
                
                if len(groups) > 1:  # Need at least 2 groups for ANOVA
                    f_val, p_val = stats.f_oneway(*groups)
                    
                    if p_val < 0.05:  # If significant
                        # Run post-hoc Tukey test
                        posthoc = pairwise_tukeyhsd(
                            region_data['Mean_Thickness'],
                            region_data['group'],
                            alpha=0.05
                        )
                        
                        significant_regions.append({
                            'Region': region,
                            'F_value': f_val,
                            'P_value': p_val,
                            'Posthoc': posthoc
                        })
            except Exception as e:
                print(f"Error analyzing region {region}: {e}")
    
    # Save significant results to CSV
    if significant_regions:
        sig_results = pd.DataFrame([(r['Region'], r['F_value'], r['P_value']) 
                                   for r in significant_regions],
                                 columns=['Region', 'F_value', 'P_value'])
        sig_results = sig_results.sort_values('P_value')
        sig_results.to_csv(os.path.join(output_path, 'significant_regions.csv'), index=False)
        
        # Plot for significant regions
        top_sig_regions = sig_results.head(10)['Region'].tolist()
        
        plt.figure(figsize=(15, 10))
        sig_regions_data = pivot_data.loc[top_sig_regions].reset_index()
        sig_regions_long = pd.melt(sig_regions_data, id_vars=['Region'], 
                                 value_vars=['HC', 'PIGD', 'TDPD'],
                                 var_name='Group', value_name='Mean Thickness')
        
        g = sns.catplot(x='Region', y='Mean Thickness', hue='Group', data=sig_regions_long,
                      kind='bar', height=8, aspect=1.5, palette=COLORS)
        g.set_xticklabels(rotation=90)
        plt.title('Top 10 Regions with Statistically Significant Group Differences', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'significant_regions_comparison.png'), dpi=300, bbox_inches='tight')
    
    print("Group comparison visualizations saved.")


def create_regional_thickness_profiles(data, output_path):
    """Create thickness profiles for different brain regions"""
    print("Generating regional thickness profiles...")
    
    # Select key functional regions of interest
    functional_groups = {
        'Motor': ['Precentral Gyrus', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
                 'Postcentral Gyrus'],
        'Cognitive': ['Frontal Pole', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus',
                     'Inferior Frontal Gyrus, pars triangularis'],
        'Limbic': ['Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division',
                  'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division'],
        'Visual': ['Lingual Gyrus', 'Occipital Pole', 'Lateral Occipital Cortex, superior division',
                  'Lateral Occipital Cortex, inferior division']
    }
    
    # Plot thickness profiles by functional group
    for func_name, regions in functional_groups.items():
        # Filter data for these regions and calculate means
        func_data = data[data['Region'].isin(regions)]
        means_by_group = func_data.groupby(['group', 'Region'])['Mean_Thickness'].mean().reset_index()
        
        # Create radar chart
        groups = means_by_group['group'].unique()
        
        # Set up the figure with appropriate size
        fig = plt.figure(figsize=(12, 10))
        
        # Create separate subplots for each group
        for i, group in enumerate(groups):
            group_data = means_by_group[means_by_group['group'] == group]
            
            # Sort by region name for consistency
            group_data = group_data.sort_values('Region')
            
            # Get thickness values and region names
            values = group_data['Mean_Thickness'].tolist()
            regions_list = group_data['Region'].tolist()
            
            # Close the loop for the radar chart
            values.append(values[0])
            regions_list.append(regions_list[0])
            
            # Calculate angles for radar chart
            angles = np.linspace(0, 2*np.pi, len(regions_list), endpoint=True)
            
            # Create subplot
            ax = fig.add_subplot(1, 3, i+1, polar=True)
            
            # Plot data
            ax.plot(angles, values, 'o-', linewidth=2, label=group, color=COLORS.get(group, 'gray'))
            ax.fill(angles, values, alpha=0.25, color=COLORS.get(group, 'gray'))
            
            # Add region labels
            ax.set_thetagrids(angles[:-1] * 180/np.pi, [r.split(',')[0] for r in regions_list[:-1]])
            
            # Add title
            ax.set_title(f'{group} - {func_name} Regions', fontsize=14)
            
            # Set y-limits (thickness scale)
            ax.set_ylim(0, max(means_by_group['Mean_Thickness'])*1.2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_path, f'{func_name}_regions_radar.png'), dpi=300, bbox_inches='tight')
    
    # Create bar charts for each functional group
    for func_name, regions in functional_groups.items():
        func_data = data[data['Region'].isin(regions)]
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Region', y='Mean_Thickness', hue='group', data=func_data, palette=COLORS)
        plt.title(f'Thickness Comparison in {func_name} Regions', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Group')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{func_name}_regions_barchart.png'), dpi=300, bbox_inches='tight')
    
    print("Regional thickness profiles saved.")


def create_subject_visualizations(data, output_path):
    """Create visualizations for individual subject thickness patterns"""
    print("Generating individual subject visualizations...")
    
    # Select a subset of key regions for clarity
    key_regions = [
        'Frontal Pole', 'Precentral Gyrus', 'Postcentral Gyrus', 'Superior Frontal Gyrus', 
        'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division',
        'Superior Parietal Lobule', 'Occipital Pole'
    ]
    
    # Filter data for key regions
    key_region_data = data[data['Region'].isin(key_regions)]
    
    # Get some representative subjects from each group
    rep_subjects = []
    
    for group in ['HC', 'PIGD', 'TDPD']:
        # Get subjects in this group
        group_subjects = key_region_data[key_region_data['group'] == group]['Subject'].unique()
        
        if len(group_subjects) > 0:
            # Take up to 3 subjects from each group
            rep_subjects.extend(group_subjects[:min(3, len(group_subjects))])
    
    # Create individual subject plots
    for subject in rep_subjects:
        # Get subject data for key regions
        subject_data = key_region_data[key_region_data['Subject'] == subject]
        
        if not subject_data.empty:
            group = subject_data['group'].iloc[0]
            
            # Create bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.bar(subject_data['Region'], subject_data['Mean_Thickness'], 
                         color=COLORS.get(group, 'gray'))
            
            # Add average line for the subject's group
            group_means = key_region_data[key_region_data['group'] == group].groupby('Region')['Mean_Thickness'].mean()
            for i, region in enumerate(subject_data['Region']):
                if region in group_means.index:
                    plt.axhline(y=group_means[region], xmin=i/len(subject_data['Region']), 
                               xmax=(i+1)/len(subject_data['Region']), 
                               color='red', linestyle='--', alpha=0.7)
            
            plt.title(f'Cortical Thickness Profile - Subject {subject} (Group: {group})', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Mean Thickness', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'subject_{subject}_profile.png'), dpi=300, bbox_inches='tight')
    
    # Create dimensionality reduction plot to show clustering of subjects
    region_pivot = data.pivot_table(index='Subject', columns='Region', values='Mean_Thickness')
    
    # Fill missing values with column means
    region_pivot = region_pivot.fillna(region_pivot.mean())
    
    # Add group information
    subject_groups = data[['Subject', 'group']].drop_duplicates().set_index('Subject')
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(region_pivot.values))
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=region_pivot.index)
    pca_df = pca_df.join(subject_groups)
    
    # Create PCA plot
    plt.figure(figsize=(12, 10))
    for group, color in COLORS.items():
        if group in ['HC', 'PIGD', 'TDPD']:
            group_data = pca_df[pca_df['group'] == group]
            plt.scatter(group_data['PC1'], group_data['PC2'], c=color, label=group, 
                      alpha=0.7, edgecolors='w', s=100)
    
    plt.title('PCA of Subject Thickness Profiles', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
    plt.legend(title='Group')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'subject_pca_clustering.png'), dpi=300, bbox_inches='tight')
    
    print("Individual subject visualizations saved.")


def create_interactive_visualizations(data, output_path):
    """Create interactive visualizations using Plotly"""
    print("Generating interactive visualizations...")
    
    # Interactive boxplot of thickness by group and region
    # Select top 10 regions with most variation
    region_std = data.groupby('Region')['Mean_Thickness'].std().nlargest(10).index
    top_region_data = data[data['Region'].isin(region_std)]
    
    fig = px.box(top_region_data, x='Region', y='Mean_Thickness', color='group',
                color_discrete_map=COLORS, points='all',
                title='Cortical Thickness Distribution by Group and Region')
    
    fig.update_layout(
        xaxis_title='Brain Region',
        yaxis_title='Mean Thickness',
        legend_title='Group',
        font=dict(size=14),
        height=800,
        width=1200
    )
    
    pio.write_html(fig, file=os.path.join(output_path, 'interactive_boxplot.html'))
    
    # Interactive 3D scatter plot for PCA
    region_pivot = data.pivot_table(index='Subject', columns='Region', values='Mean_Thickness')
    
    # Fill missing values with column means
    region_pivot = region_pivot.fillna(region_pivot.mean())
    
    # Add group information
    subject_groups = data[['Subject', 'group']].drop_duplicates().set_index('Subject')
    
    # Perform PCA with 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(region_pivot.values))
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'], index=region_pivot.index)
    pca_df = pca_df.join(subject_groups).reset_index()
    
    # Create 3D scatter plot
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                      color='group', color_discrete_map=COLORS,
                      hover_name='Subject',
                      title='3D PCA of Cortical Thickness Profiles')
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
        ),
        width=1000,
        height=800
    )
    
    pio.write_html(fig, file=os.path.join(output_path, 'interactive_3d_pca.html'))
    
    # Interactive heatmap of regional thickness
    pivot_data = data.pivot_table(
        index='Region', 
        columns='group', 
        values='Mean_Thickness', 
        aggfunc='mean'
    )
    
    # Calculate the difference between groups
    pivot_data['HC-PIGD'] = pivot_data['HC'] - pivot_data['PIGD']
    pivot_data['HC-TDPD'] = pivot_data['HC'] - pivot_data['TDPD']
    pivot_data['PIGD-TDPD'] = pivot_data['PIGD'] - pivot_data['TDPD']
    
    # Sort by the absolute maximum difference between any groups
    pivot_data['max_abs_diff'] = pivot_data[['HC-PIGD', 'HC-TDPD', 'PIGD-TDPD']].abs().max(axis=1)
    pivot_data = pivot_data.sort_values('max_abs_diff', ascending=False)
    
    # Take top 20 regions
    top_diff_regions = pivot_data.head(20).index.tolist()
    
    # Create heatmap of top regions
    heatmap_data = data[data['Region'].isin(top_diff_regions)]
    
    fig = px.density_heatmap(
        heatmap_data, 
        x='Region', 
        y='group', 
        z='Mean_Thickness',
        title='Mean Cortical Thickness by Group and Region<br>(Top 20 Regions with Greatest Group Differences)'
    )
    
    fig.update_layout(
        xaxis_title='Brain Region',
        yaxis_title='Group',
        xaxis={'categoryorder': 'array', 'categoryarray': top_diff_regions},
        font=dict(size=14),
        height=800,
        width=1200
    )
    
    pio.write_html(fig, file=os.path.join(output_path, 'interactive_heatmap.html'))
    
    print("Interactive visualizations saved.")


def create_hemisphere_comparison(data, output_path):
    """Create visualizations comparing left and right hemisphere thickness"""
    print("Generating hemisphere comparison visualizations...")
    
    # Identify hemisphere from region names (if available)
    def get_hemisphere(region_name):
        if ', left' in region_name.lower():
            return 'Left'
        elif ', right' in region_name.lower():
            return 'Right'
        else:
            return 'Unknown'
    
    # Check if we have hemisphere information
    data['Hemisphere'] = data['Region'].apply(get_hemisphere)
    
    # If we have hemisphere information
    if len(data[data['Hemisphere'] != 'Unknown']) > 0:
        # Create boxplots comparing hemispheres by group
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='group', y='Mean_Thickness', hue='Hemisphere', data=data[data['Hemisphere'] != 'Unknown'], 
                  palette={'Left': '#1f77b4', 'Right': '#ff7f0e'})
        plt.title('Hemisphere Cortical Thickness Comparison by Group', fontsize=16)
        plt.xlabel('Group', fontsize=14)
        plt.ylabel('Mean Thickness', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'hemisphere_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Create hemisphere asymmetry visualization
        # For each subject and region, calculate left-right difference
        if len(data[data['Hemisphere'] == 'Left']) > 0 and len(data[data['Hemisphere'] == 'Right']) > 0:
            # Get base region name (without hemisphere designation)
            data['BaseRegion'] = data['Region'].apply(lambda x: x.replace(', left', '').replace(', right', ''))
            
            # Pivot to get left and right values side by side
            asymmetry_data = data[data['Hemisphere'] != 'Unknown'].pivot_table(
                index=['Subject', 'group', 'BaseRegion'], 
                columns='Hemisphere', 
                values='Mean_Thickness'
            )
            
            # Calculate asymmetry (Left - Right)
            asymmetry_data['Asymmetry'] = asymmetry_data['Left'] - asymmetry_data['Right']
            asymmetry_data = asymmetry_data.reset_index()
            
            # Create boxplots of asymmetry by group
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='group', y='Asymmetry', data=asymmetry_data, palette=COLORS)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.title('Hemisphere Asymmetry by Group (Left - Right)', fontsize=16)
            plt.xlabel('Group', fontsize=14)
            plt.ylabel('Thickness Asymmetry (Left - Right)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'hemisphere_asymmetry.png'), dpi=300, bbox_inches='tight')
            
            # Find regions with greatest asymmetry
            region_asymmetry = asymmetry_data.groupby(['group', 'BaseRegion'])['Asymmetry'].agg(['mean', 'std']).reset_index()
            region_asymmetry['abs_mean'] = region_asymmetry['mean'].abs()
            top_asymmetry_regions = region_asymmetry.nlargest(10, 'abs_mean')
            
            # Plot top asymmetry regions
            plt.figure(figsize=(16, 10))
            sns.barplot(x='BaseRegion', y='mean', hue='group', data=top_asymmetry_regions, palette=COLORS)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.title('Regions with Greatest Hemispheric Asymmetry', fontsize=16)
            plt.xlabel('Region', fontsize=14)
            plt.ylabel('Mean Asymmetry (Left - Right)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'top_asymmetry_regions.png'), dpi=300, bbox_inches='tight')
    
    print("Hemisphere comparison visualizations saved.")


def create_network_analysis(data, output_path):
    """Create brain network visualizations based on cortical thickness correlations"""
    print("Generating network analysis visualizations...")
    
    # Create correlation matrix of thickness across regions
    network_pivot = data.pivot_table(index='Subject', columns='Region', values='Mean_Thickness')
    correlation_matrix = network_pivot.corr(method='pearson')
    
    # Create heatmap of correlations
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1,
               square=True, linewidths=0.5, annot=False)
    plt.title('Correlation Matrix of Regional Cortical Thickness', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'thickness_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Create separate correlation matrices for each group
    for group in ['HC', 'PIGD', 'TDPD']:
        # Filter data for this group
        group_data = data[data['group'] == group]
        
        if len(group_data) > 0:
            # Create pivot table and correlation matrix
            group_pivot = group_data.pivot_table(index='Subject', columns='Region', values='Mean_Thickness')
            
            # Only proceed if we have enough data
            if group_pivot.shape[0] > 1:
                group_corr = group_pivot.corr(method='pearson')
                
                # Create heatmap
                plt.figure(figsize=(16, 14))
                mask = np.triu(np.ones_like(group_corr, dtype=bool))
                sns.heatmap(group_corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1,
                          square=True, linewidths=0.5, annot=False)
                plt.title(f'Correlation Matrix - {group} Group', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f'{group}_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Find top correlated region pairs
    corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            region1 = correlation_matrix.columns[i]
            region2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            
            corr_pairs.append({
                'Region1': region1,
                'Region2': region2,
                'Correlation': corr_value,
                'AbsCorrelation': abs(corr_value)
            })
    
    # Convert to DataFrame and sort
    corr_pairs_df = pd.DataFrame(corr_pairs)
    top_positive_pairs = corr_pairs_df.nlargest(15, 'Correlation')
    top_negative_pairs = corr_pairs_df.nsmallest(15, 'Correlation')
    
    # Plot top positive correlations
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Correlation', y='Region1 + Region2', 
              data=top_positive_pairs.assign(**{'Region1 + Region2': top_positive_pairs['Region1'] + ' & ' + top_positive_pairs['Region2']}),
              color='blue')
    plt.title('Top 15 Positive Regional Thickness Correlations', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'top_positive_correlations.png'), dpi=300, bbox_inches='tight')
    
    # Plot top negative correlations
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Correlation', y='Region1 + Region2', 
              data=top_negative_pairs.assign(**{'Region1 + Region2': top_negative_pairs['Region1'] + ' & ' + top_negative_pairs['Region2']}),
              color='red')
    plt.title('Top 15 Negative Regional Thickness Correlations', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'top_negative_correlations.png'), dpi=300, bbox_inches='tight')
    
    print("Network analysis visualizations saved.")


def generate_html_report(output_path):
    """Generate an HTML report that combines all the visualizations"""
    print("Generating HTML report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cortical Thickness Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2C7BB6;
                border-bottom: 2px solid #2C7BB6;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #D7191C;
                margin-top: 30px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .image-container {{
                text-align: center;
            }}
            .caption {{
                font-style: italic;
                margin-top: 5px;
                color: #666;
            }}
            .interactive-link {{
                display: block;
                margin: 20px 0;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
                text-decoration: none;
                color: #2C7BB6;
                font-weight: bold;
            }}
            .interactive-link:hover {{
                background-color: #e0e0e0;
            }}
        </style>
    </head>
    <body>
        <h1>Cortical Thickness Analysis Report</h1>
        
        <div class="section">
            <h2>Brain Region Visualizations</h2>
            <p>These visualizations show the distribution of cortical thickness across different brain regions.</p>
            
            <div class="image-container">
                <img src="brain_regions_thickness_map.png" alt="Brain Regions Thickness Map">
                <p class="caption">Brain regions colored by cortical thickness.</p>
            </div>
            
            <div class="image-container">
                <img src="glass_brain_regions.png" alt="Glass Brain Regions">
                <p class="caption">Glass brain visualization of cortical regions.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Group Comparison Analysis</h2>
            <p>These visualizations compare cortical thickness across the three subject groups: HC (Healthy Controls), PIGD (Postural Instability and Gait Disorder), and TDPD (Tremor-Dominant Parkinson's Disease).</p>
            
            <div class="image-container">
                <img src="group_thickness_boxplot.png" alt="Group Thickness Boxplot">
                <p class="caption">Distribution of cortical thickness by group.</p>
            </div>
            
            <div class="image-container">
                <img src="group_region_thickness_heatmap.png" alt="Group Region Thickness Heatmap">
                <p class="caption">Heatmap of regional thickness by group.</p>
            </div>
            
            <div class="image-container">
                <img src="top_regions_group_comparison.png" alt="Top Regions Group Comparison">
                <p class="caption">Top regions with greatest differences between groups.</p>
            </div>
            
            <div class="image-container">
                <img src="significant_regions_comparison.png" alt="Significant Regions Comparison">
                <p class="caption">Regions with statistically significant group differences.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Functional Region Profiles</h2>
            <p>These visualizations show thickness profiles across different functional brain regions.</p>
            
            <div class="image-container">
                <img src="Motor_regions_radar.png" alt="Motor Regions Radar">
                <p class="caption">Thickness profile of motor regions across groups.</p>
            </div>
            
            <div class="image-container">
                <img src="Cognitive_regions_radar.png" alt="Cognitive Regions Radar">
                <p class="caption">Thickness profile of cognitive regions across groups.</p>
            </div>
            
            <div class="image-container">
                <img src="Limbic_regions_radar.png" alt="Limbic Regions Radar">
                <p class="caption">Thickness profile of limbic regions across groups.</p>
            </div>
            
            <div class="image-container">
                <img src="Visual_regions_radar.png" alt="Visual Regions Radar">
                <p class="caption">Thickness profile of visual regions across groups.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Individual Subject Analysis</h2>
            <p>These visualizations show thickness patterns for individual subjects and how they cluster together.</p>
            
            <div class="image-container">
                <img src="subject_pca_clustering.png" alt="Subject PCA Clustering">
                <p class="caption">PCA clustering of subjects based on thickness profiles.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Hemisphere Comparison</h2>
            <p>These visualizations compare thickness between left and right hemispheres.</p>
            
            <div class="image-container">
                <img src="hemisphere_comparison.png" alt="Hemisphere Comparison">
                <p class="caption">Comparison of thickness between left and right hemispheres by group.</p>
            </div>
            
            <div class="image-container">
                <img src="hemisphere_asymmetry.png" alt="Hemisphere Asymmetry">
                <p class="caption">Hemispheric asymmetry by group.</p>
            </div>
            
            <div class="image-container">
                <img src="top_asymmetry_regions.png" alt="Top Asymmetry Regions">
                <p class="caption">Regions with greatest hemispheric asymmetry.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Network Analysis</h2>
            <p>These visualizations show network relationships between different brain regions based on thickness correlations.</p>
            
            <div class="image-container">
                <img src="thickness_correlation_matrix.png" alt="Thickness Correlation Matrix">
                <p class="caption">Correlation matrix of regional thickness.</p>
            </div>
            
            <div class="image-container">
                <img src="HC_correlation_matrix.png" alt="HC Correlation Matrix">
                <p class="caption">Correlation matrix for HC group.</p>
            </div>
            
            <div class="image-container">
                <img src="PIGD_correlation_matrix.png" alt="PIGD Correlation Matrix">
                <p class="caption">Correlation matrix for PIGD group.</p>
            </div>
            
            <div class="image-container">
                <img src="TDPD_correlation_matrix.png" alt="TDPD Correlation Matrix">
                <p class="caption">Correlation matrix for TDPD group.</p>
            </div>
            
            <div class="image-container">
                <img src="top_positive_correlations.png" alt="Top Positive Correlations">
                <p class="caption">Top positive correlations between regions.</p>
            </div>
            
            <div class="image-container">
                <img src="top_negative_correlations.png" alt="Top Negative Correlations">
                <p class="caption">Top negative correlations between regions.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Interactive Visualizations</h2>
            <p>Click the links below to open interactive visualizations in your web browser.</p>
            
            <a href="interactive_boxplot.html" class="interactive-link">Interactive Regional Thickness Boxplot</a>
            <a href="interactive_3d_pca.html" class="interactive-link">Interactive 3D PCA of Thickness Profiles</a>
            <a href="interactive_heatmap.html" class="interactive-link">Interactive Regional Thickness Heatmap</a>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_path, 'thickness_report.html'), 'w') as f:
        f.write(html_content)
    
    print("HTML report generated.")


def main():
    """Main function to run the analysis pipeline"""
    print("Starting cortical thickness visualization pipeline...")
    
    # Create output directory
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    print(f"Output directory: {VISUALIZATION_DIR}")
    
    # Load data
    data, group_stats, subject_info = load_data()
    
    # Generate visualizations
    plot_brain_regions_thickness(data, VISUALIZATION_DIR)
    plot_group_comparisons(data, group_stats, VISUALIZATION_DIR)
    create_regional_thickness_profiles(data, VISUALIZATION_DIR)
    create_subject_visualizations(data, VISUALIZATION_DIR)
    create_hemisphere_comparison(data, VISUALIZATION_DIR)
    create_network_analysis(data, VISUALIZATION_DIR)
    create_interactive_visualizations(data, VISUALIZATION_DIR)
    
    # Generate HTML report
    generate_html_report(VISUALIZATION_DIR)
    
    print("Visualization pipeline complete!")
    print(f"Results saved to: {VISUALIZATION_DIR}")
    print("Open thickness_report.html in a web browser to view the complete report.")


if __name__ == "__main__":
    main()