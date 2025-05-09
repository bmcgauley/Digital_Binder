#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Comprehensive PCA Analysis

This script performs Principal Component Analysis (PCA) on multiple datasets,
demonstrating the standard steps and visualizations in a typical PCA workflow.

The script is structured to handle three datasets:
- Wine quality dataset
- Economic data
- City statistics

For each dataset, we'll create a set of standardized PCA visualizations and analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import time
import sys

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create a markdown-style section header function
def print_section(title, level=1):
    """Print a markdown-style section header"""
    if level == 1:
        print(f"\n{'='*80}\n## {title}\n{'='*80}")
    elif level == 2:
        print(f"\n{'-'*80}\n### {title}\n{'-'*80}")
    else:
        print(f"\n#### {title}")

# Function to create confidence ellipse for 2D PCA plot
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Main function for PCA analysis
def perform_pca_analysis(dataset_path, dataset_name, target_variable=None, categorical_threshold=10):
    """
    Perform comprehensive PCA analysis on a dataset
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    dataset_name : str
        Name of the dataset (for output folder)
    target_variable : str, optional
        Name of the target variable (if any)
    categorical_threshold : int, optional
        Maximum number of unique values for a column to be considered categorical
    """
    charts_dir = f"charts/{dataset_name}"
    os.makedirs(charts_dir, exist_ok=True)
    
    print_section(f"PCA Analysis for {dataset_name} Dataset")
    
    # Phase 1: Data Loading and Exploration
    print_section("Phase 1: Data Loading and Exploration", level=2)
    print("Loading dataset and performing initial exploration...")
    
    try:
        # Load data with proper handling for mixed types
        df = pd.read_csv(dataset_path, low_memory=False)
        print(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Display basic information
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Column information
    print("\nColumn names and data types:")
    for i, (col, dtype) in enumerate(df.dtypes.items()):
        print(f"{i+1}. {col} ({dtype})")
    
    # Identify target variable if provided
    if target_variable and target_variable in df.columns:
        print(f"\nTarget variable '{target_variable}' statistics:")
        try:
            print(f"Mean: {df[target_variable].mean():.2f}")
            print(f"Min: {df[target_variable].min():.2f}")
            print(f"Max: {df[target_variable].max():.2f}")
        except:
            print("Cannot compute statistics for non-numeric target variable")
            if df[target_variable].nunique() < 20:
                print("Value counts:")
                print(df[target_variable].value_counts())
    
    # Handle missing values - show counts
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")
    
    # Phase 2: Data Cleaning and Preparation
    print_section("Phase 2: Data Cleaning and Preparation", level=2)
    print("Cleaning data and preparing for PCA...")
    
    # Handle missing values in the target variable
    df_clean = df.copy()
    if target_variable and target_variable in df.columns:
        df_clean = df_clean.dropna(subset=[target_variable])
        print(f"Rows after dropping target nulls: {len(df_clean)} (removed {len(df) - len(df_clean)} rows)")
    
    # Identify columns with high percentage of missing values
    drop_cols = []
    for col in df_clean.columns:
        # If more than 50% values are missing, drop the column
        if df_clean[col].isnull().mean() > 0.5:
            drop_cols.append(col)
    
    if drop_cols:
        print(f"\nDropping {len(drop_cols)} columns with too many missing values:")
        print(drop_cols)
        df_clean = df_clean.drop(columns=drop_cols)
    
    print(f"\nClean dataset shape: {df_clean.shape}")
    
    # Create a histogram of the target variable if numeric
    if target_variable and target_variable in df_clean.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_clean[target_variable], kde=True)
            plt.title(f"Distribution of {target_variable}")
            plt.xlabel(target_variable)
            plt.tight_layout()
            plt.savefig(f"{charts_dir}/target_distribution.png")
            print(f"Target distribution plot saved as '{charts_dir}/target_distribution.png'")
        except:
            print(f"Could not create histogram for non-numeric target: {target_variable}")
    
    # Identify numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_variable and target_variable in numeric_cols:
        numeric_cols.remove(target_variable)
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Check if any numeric columns should be treated as categorical
    for col in list(numeric_cols):
        if df_clean[col].nunique() < categorical_threshold:
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    print(f"\nNumeric columns identified: {len(numeric_cols)}")
    print(numeric_cols)
    
    print(f"\nCategorical columns identified: {len(categorical_cols)}")
    print(categorical_cols)
    
    # Handle date-like columns (specific preprocessing for date columns)
    date_components = {}
    for col in list(categorical_cols):
        # Check if column might be a date (e.g., contains period or dates)
        if 'date' in col.lower() or 'period' in col.lower() or 'year' in col.lower():
            try:
                # Try to split by common date separators
                if df_clean[col].astype(str).str.contains('[-./]').any():
                    # This is a date with separators
                    components = df_clean[col].astype(str).str.split('[-./]', expand=True)
                    date_components[col] = components
                    
                    # Create new columns for each component
                    for i in range(components.shape[1]):
                        new_col = f"{col}_component_{i}"
                        df_clean[new_col] = components[i].astype(float)
                        numeric_cols.append(new_col)
                    
                    categorical_cols.remove(col)
                # Check for period format (e.g., 2020.03)
                elif df_clean[col].astype(str).str.contains('[.]').any():
                    components = df_clean[col].astype(str).str.split('.', expand=True)
                    date_components[col] = components
                    
                    # If format looks like year.month
                    if len(components.columns) == 2:
                        df_clean[f"{col}_year"] = components[0].astype(float)
                        df_clean[f"{col}_month"] = components[1].astype(float)
                        numeric_cols.extend([f"{col}_year", f"{col}_month"])
                        categorical_cols.remove(col)
            except:
                print(f"Could not process date-like column: {col}")
    
    # Sample the data if it's very large
    sample_size = min(50000, len(df_clean))
    df_sample = df_clean.sample(sample_size, random_state=42)
    
    # Phase 3: Correlation Analysis
    print_section("Phase 3: Correlation Analysis", level=2)
    print("Analyzing correlations between features...")
    
    # Create a correlation matrix of numeric features
    if numeric_cols:
        print("\nCreating correlation matrix of numeric features...")
        plt.figure(figsize=(12, 10))
        numeric_for_corr = numeric_cols.copy()
        
        # If there are too many numeric columns, select a subset
        if len(numeric_for_corr) > 15:
            numeric_for_corr = numeric_for_corr[:15]  # Take first 15 columns
            print(f"Too many numeric columns, using only the first 15 for correlation matrix")
        
        # Add target variable for correlation analysis
        if target_variable and target_variable in df_clean.columns:
            if df_clean[target_variable].dtype in (np.float64, np.int64):
                numeric_for_corr.append(target_variable)
        
        corr_matrix = df_sample[numeric_for_corr].corr()
        
        # Plot the correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                    square=True, linewidths=.5)
        plt.title("Numeric Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/correlation_matrix_numeric.png")
        print(f"Correlation matrix saved as '{charts_dir}/correlation_matrix_numeric.png'")
    
    # Phase 4: Feature Engineering and Selection
    print_section("Phase 4: Feature Engineering and Selection", level=2)
    print("Preparing final feature set for PCA...")
    
    # Start with numeric features
    X_features = numeric_cols.copy()
    
    # Limit categorical columns to avoid feature explosion (focus on the most important ones)
    key_categorical = categorical_cols[:3] if len(categorical_cols) > 3 else categorical_cols
    
    # Create feature dataframe
    X = df_sample[X_features].copy() if X_features else pd.DataFrame(index=df_sample.index)
    
    # Add encoded categorical columns
    print("\nEncoding categorical features...")
    for cat_col in key_categorical:
        if cat_col in df_sample.columns:  # Verify column exists
            col_name = f"{cat_col}_Code"
            X[col_name] = df_sample[cat_col].astype('category').cat.codes
            print(f"Encoded '{cat_col}' as '{col_name}'")
    
    # Fill missing values
    print("\nFilling any remaining missing values...")
    if not X.empty:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    print("\nFinal features for PCA:")
    for i, col in enumerate(X.columns):
        print(f"{i+1}. {col}")
    print(f"\nFeature dataframe shape: {X.shape}")
    
    # Phase 5: PCA Analysis
    print_section("Phase 5: PCA Analysis", level=2)
    
    # Skip PCA if no features are available
    if X.empty or X.shape[1] < 2:
        print("Not enough features for PCA analysis. At least 2 features are required.")
        return
    
    print("Performing Principal Component Analysis...")
    
    # Create final correlation matrix with selected features
    plt.figure(figsize=(12, 10))
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                square=True, linewidths=.5)
    plt.title("Final Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/correlation_matrix_final.png")
    print(f"Final correlation matrix saved as '{charts_dir}/correlation_matrix_final.png'")
    
    # Scale the data
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    print("\nRunning PCA algorithm...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    print("PCA completed successfully!")
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find how many components explain key variance thresholds
    variance_thresholds = [0.7, 0.8, 0.9, 0.95]
    for threshold in variance_thresholds:
        components_needed = np.argmax(cumulative_variance >= threshold) + 1
        print(f"Components needed for {threshold*100:.0f}% variance: {components_needed}")
    
    # Phase 6: PCA Visualizations
    print_section("Phase 6: PCA Visualizations", level=2)
    print("Creating PCA visualizations...")
    
    # 1. Scree Plot with Kaiser criterion
    print("\nCreating scree plot...")
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7,
            label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
             label='Cumulative explained variance', color='red')
    
    # Add Kaiser criterion line (eigenvalues > 1)
    kaiser_mask = pca.explained_variance_ > 1.0
    kaiser_components = sum(kaiser_mask)
    if kaiser_components > 0:
        plt.axhline(y=1/len(pca.explained_variance_), color='orange', linestyle='--',
                   label='Kaiser criterion')
        plt.axvline(x=kaiser_components, color='green', linestyle=':',
                   label=f'{kaiser_components} components (Kaiser)')
    
    # Add threshold line
    main_threshold = 0.8  # We'll highlight the 80% threshold
    plt.axhline(y=main_threshold, color='blue', linestyle='-.',
               label=f'{main_threshold*100:.0f}% variance threshold')
    components_for_main = np.argmax(cumulative_variance >= main_threshold) + 1
    plt.axvline(x=components_for_main, color='purple', linestyle='-.',
               label=f'{components_for_main} components needed')
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot: Explained Variance by Components')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/pca_scree_plot.png")
    print(f"Scree plot saved as '{charts_dir}/pca_scree_plot.png'")
    
    # 2. Feature Loadings Visualization
    print("\nAnalyzing feature importance in principal components...")
    feature_names = X.columns
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_names
    )
    
    # Display loadings for the first few components
    n_display = min(3, len(pca.components_))
    print("\nFeature loadings for the first few principal components:")
    for i in range(n_display):
        pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(f"\nPC{i+1} feature importance (absolute values):")
        for feat, value in pc_loadings.head(5).items():  # Show top 5
            print(f"{feat}: {value:.4f} ({loadings.loc[feat, f'PC{i+1}']:.4f})")
    
    # Create heatmap of loadings
    n_components_to_plot = min(5, len(pca.components_))
    n_features_to_plot = min(10, len(feature_names))
    
    plt.figure(figsize=(12, 8))
    loadings_subset = loadings.iloc[:n_features_to_plot, :n_components_to_plot]
    sns.heatmap(loadings_subset, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={"shrink": .8})
    plt.title(f"Feature Loadings in First {n_components_to_plot} Principal Components")
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/feature_loadings_heatmap.png")
    print(f"Feature loadings heatmap saved as '{charts_dir}/feature_loadings_heatmap.png'")
    
    # 3. Feature Importance Bar Charts for PC1 and PC2
    plt.figure(figsize=(14, 6))
    
    # Sort loadings for PC1 and PC2
    pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
    pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
    
    # Plot PC1 importance
    plt.subplot(1, 2, 1)
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in loadings.loc[pc1_loadings.index, 'PC1']]
    pc1_loadings.head(5).plot(kind='bar', color=colors)
    plt.title('PC1 Top Feature Contributions')
    plt.ylabel('Absolute Loading Value')
    plt.xticks(rotation=45, ha='right')
    
    # Plot PC2 importance
    plt.subplot(1, 2, 2)
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in loadings.loc[pc2_loadings.index, 'PC2']]
    pc2_loadings.head(5).plot(kind='bar', color=colors)
    plt.title('PC2 Top Feature Contributions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/pc_feature_importance.png")
    print(f"PC feature importance plots saved as '{charts_dir}/pc_feature_importance.png'")
    
    # 4. Biplot (PCA with feature vector overlay)
    print("\nCreating PCA biplot...")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot of samples
    if target_variable and target_variable in df_sample.columns:
        try:
            target_values = df_sample[target_variable].values
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_values, 
                                 cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label=target_variable)
        except:
            # If target is categorical or non-numeric
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
    
    # Add feature vectors
    if len(pca.components_) >= 2 and len(feature_names) > 0:
        feature_vectors = pca.components_[:2].T
        # Scale the vectors for visibility
        scale = 5
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, 
                     feature_vectors[i, 0] * scale, 
                     feature_vectors[i, 1] * scale, 
                     head_width=0.2, head_length=0.2, fc='r', ec='r', alpha=0.5)
            plt.text(feature_vectors[i, 0] * scale * 1.1, 
                    feature_vectors[i, 1] * scale * 1.1, 
                    feature, color='r')
    
    # Add confidence ellipse
    if X_pca.shape[0] > 10:  # Only if we have enough samples
        confidence_ellipse(X_pca[:, 0], X_pca[:, 1], plt.gca(), 
                         n_std=2.0, edgecolor='darkblue', linestyle='--', 
                         label='95% Confidence Ellipse')
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel(f'First Principal Component ({explained_variance[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({explained_variance[1]:.1%} variance)')
    plt.title('PCA Biplot: Samples and Feature Vectors')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/pca_biplot.png")
    print(f"PCA biplot saved as '{charts_dir}/pca_biplot.png'")
    
    # 5. 3D Plot (if we have at least 3 components)
    if len(pca.components_) >= 3:
        print("\nCreating 3D PCA visualization...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the samples
        if target_variable and target_variable in df_sample.columns:
            try:
                target_values = df_sample[target_variable].values
                p = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                              c=target_values, cmap='viridis', alpha=0.6, s=20)
                plt.colorbar(p, ax=ax, label=target_variable)
            except:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6, s=20)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6, s=20)
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
        ax.set_title('3D PCA Visualization')
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/pca_3d.png")
        print(f"3D PCA visualization saved as '{charts_dir}/pca_3d.png'")
    
    # Phase 7: Summary and Interpretation
    print_section("Phase 7: Summary and Interpretation", level=2)
    
    # Summary of PCA results
    print("\n=== PCA ANALYSIS SUMMARY ===")
    print(f"Dataset: {dataset_name}")
    print(f"Total number of components needed for 80% variance: {components_for_main}")
    
    if kaiser_components > 0:
        print(f"Components with eigenvalues > 1 (Kaiser criterion): {kaiser_components}")
    
    print("\nMost important features (based on PC1 and PC2):")
    for feat, value in pc1_loadings.head(3).items():
        print(f"PC1 - {feat}: {loadings.loc[feat, 'PC1']:.4f}")
    for feat, value in pc2_loadings.head(3).items():
        print(f"PC2 - {feat}: {loadings.loc[feat, 'PC2']:.4f}")
    
    print("\nExplained variance by component:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"PC{i+1}: {var:.2%}")
    
    print("\nThese features contribute most to the variance in your dataset.")
    print(f"For your {dataset_name} data, these are the key variables you should focus on.")
    
    print("\nPCA analysis complete! All visualizations saved in the charts directory.")
    return pca, X_pca, X, loadings


# Execute PCA on multiple datasets
if __name__ == "__main__":
    print("""
# Comprehensive PCA Analysis
This script performs Principal Component Analysis on multiple datasets.
For each dataset, we'll create visualizations and analyze the results.
    """)
    
    # Check if datasets exist
    datasets = [
        {"path": "wine.csv", "name": "wine", "target": "quality"},
        {"path": "economic1.csv", "name": "economic", "target": "Data_value"},
        {"path": "cities.csv", "name": "cities", "target": None}
    ]
    
    # Find which datasets are available
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset["path"]):
            available_datasets.append(dataset)
        else:
            print(f"Warning: {dataset['path']} not found. Skipping this dataset.")
    
    if not available_datasets:
        print("Error: No datasets found. Please make sure at least one dataset exists.")
        sys.exit(1)
    
    print(f"Found {len(available_datasets)} datasets to analyze.")
    
    # Run PCA on each available dataset
    for dataset in available_datasets:
        print("\n" + "="*80)
        print(f"Processing dataset: {dataset['name']}")
        print("="*80)
        
        # Perform PCA analysis
        try:
            pca, X_pca, X, loadings = perform_pca_analysis(
                dataset_path=dataset["path"],
                dataset_name=dataset["name"],
                target_variable=dataset["target"]
            )
            print(f"PCA analysis for {dataset['name']} completed successfully.")
        except Exception as e:
            print(f"Error analyzing {dataset['name']}: {str(e)}")
        
        print("\n" + "-"*80)
    
    print("\nAll analyses completed. Check the 'charts' directory for results.") 