# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.gridspec import GridSpec
import os

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create charts directory if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# Step 1: Load the data
print("\nStep 1: Loading the data...")
try:
    wine_df = pd.read_csv("wine.csv")
    print(f"Dataset loaded with {wine_df.shape[0]} rows and {wine_df.shape[1]} columns")
except FileNotFoundError:
    print("Please ensure the wine.csv file is in the current directory")
    exit()

# Step 2: Explore the data
print("\nStep 2: Exploring the data...")
print("\nFirst few rows:")
print(wine_df.head())
print("\nDescriptive Statistics:")
print(wine_df.describe().round(2))
print("\nMissing Values:")
print(wine_df.isnull().sum())

# Step 3: Prepare features for PCA
print("\nStep 3: Preparing features...")
# Separate target if it exists
if 'class' in wine_df.columns:
    features = wine_df.drop(columns=['class'])
    target = wine_df['class']
else:
    features = wine_df.copy()

# Keep only numeric columns
features = features.select_dtypes(include=['float64', 'int64'])
print(f"Features selected for PCA: {features.columns.tolist()}")

# Step 4: Standardize the features
print("\nStep 4: Standardizing features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("Features have been standardized")

# Step 5: Perform PCA
print("\nStep 5: Performing PCA...")
pca = PCA()
transformed_features = pca.fit_transform(scaled_features)
explained_variance_ratio = pca.explained_variance_ratio_

# Step 6: Analyze explained variance
print("\nStep 6: Analyzing explained variance...")
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
threshold = 0.6

# Create the visualization
plt.figure(figsize=(15, 8))

# Plot cumulative explained variance
plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
        cumulative_variance_ratio, 
        'b-', 
        label='Cumulative explained variance')

# Plot individual explained variance
plt.bar(range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
        alpha=0.5,
        label='Individual explained variance')

# Plot threshold line
plt.axhline(y=threshold, color='r', linestyle='--', 
           label=f'{threshold*100}% variance threshold')

# Find number of components needed for threshold
n_components_needed = np.argmax(cumulative_variance_ratio >= threshold) + 1
plt.axvline(x=n_components_needed, color='g', linestyle='--',
           label=f'{n_components_needed} components needed')

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components - Wine Dataset')
plt.legend(loc='lower right')
plt.grid(True)

# Save the plot
plt.savefig('charts/explained_variance.png')
plt.close()

# Step 7: Analyze top components
print("\nStep 7: Analyzing top components...")
n_components = 3
n_top_features = 5

component_details = []
feature_names = features.columns

for i in range(n_components):
    component = pca.components_[i]
    top_indices = np.abs(component).argsort()[-n_top_features:][::-1]
    
    print(f"\nPrincipal Component {i+1}")
    print(f"Explained variance: {explained_variance_ratio[i]:.2%}")
    print("Top contributing features:")
    for idx in top_indices:
        print(f"  * {feature_names[idx]}: {component[idx]:.3f}")

# Step 8: Generate report
print("\nStep 8: Generating report...")
report = f"""
# PCA Analysis Report

## Dataset Overview
- Number of samples: {wine_df.shape[0]}
- Number of features: {features.shape[1]}
- Missing values: {wine_df.isnull().sum().sum()}

## PCA Results
- Number of components needed to explain {threshold*100}% of variance: {n_components_needed}
- Total variance explained by first 3 components: {sum(explained_variance_ratio[:3]):.2%}

## Component Breakdown
"""

for i in range(n_components):
    component = pca.components_[i]
    top_indices = np.abs(component).argsort()[-n_top_features:][::-1]
    
    report += f"\n### Principal Component {i+1}\n"
    report += f"- Explained variance: {explained_variance_ratio[i]:.2%}\n"
    report += "- Top contributing features:\n"
    for idx in top_indices:
        report += f"  * {feature_names[idx]}: {component[idx]:.3f}\n"

# Save the report
with open('pca_analysis_report.md', 'w') as f:
    f.write(report)

print("\nAnalysis complete! Check the generated report and charts in the output directory.") 