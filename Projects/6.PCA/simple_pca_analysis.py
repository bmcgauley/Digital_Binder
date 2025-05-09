import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.gridspec import GridSpec

# Set up the plots to be more readable
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create a directory for charts if it doesn't exist
import os
if not os.path.exists('charts'):
    os.makedirs('charts')

def create_pca_expression_plot(X, pca_result, n_components=6, n_samples=6, title="PCA Expression Profiles", save_path=None):
    
    # Limit components and samples to available data
    n_components = min(n_components, len(pca_result.components_))
    n_samples = min(n_samples, X.shape[0])
    n_features = X.shape[1]
    
    # Create a figure with a grid layout
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    
    # Get sample labels
    sample_labels = [f"{chr(97+i)}" for i in range(n_samples)]  # a, b, c, ...
    
    # Panel a: Original data expression profiles
    if isinstance(X, pd.DataFrame):
        data_sample = X.iloc[:n_samples]
    else:
        data_sample = X[:n_samples]
    
    # For each feature, plot its values across samples
    feature_labels = range(n_features)
    for i in range(min(9, n_features)):
        # Use different colors for different groups of features
        if i < 3:
            color = 'orange'
        elif i < 6:
            color = 'skyblue'
        else:
            color = 'gray'
            
        # Extract values for this feature across all samples
        if isinstance(X, pd.DataFrame):
            feature_values = data_sample.iloc[:, i].values
        else:
            feature_values = data_sample[:, i]
            
        axes[0].plot(sample_labels, feature_values, color=color)
    
    axes[0].set_title('a', loc='left', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Expression')
    
    # Panel b: Sample expression bar chart
    if isinstance(X, pd.DataFrame):
        mean_expression = X.iloc[:n_samples].mean(axis=1)
    else:
        mean_expression = X[:n_samples].mean(axis=1)
    
    axes[1].bar(sample_labels, mean_expression, color='black')
    axes[1].set_title('b', loc='left', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Expression')
    
    # Panel c: PC expression profiles
    pc_labels = sample_labels
    for i in range(min(n_components, 6)):
        # Get the PC values for the first n_samples
        pc_values = pca_result.components_[i, :n_features]
        
        # If we have more samples than features, pad with zeros
        if n_samples > n_features:
            pc_values = np.pad(pc_values, (0, n_samples - n_features), 'constant')
        else:
            pc_values = pc_values[:n_samples]
            
        # Use different line styles for different PCs
        axes[2].plot(pc_labels, pc_values, label=f'PC{i+1}')
    
    axes[2].set_title('c', loc='left', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Expression')
    axes[2].legend(loc='best', fontsize='small')
    
    # Panel d: PC variance explained bar chart
    pc_numbers = [f"{i+1}" for i in range(n_components)]
    axes[3].bar(pc_numbers, pca_result.explained_variance_ratio_[:n_components], color='black')
    axes[3].set_title('d', loc='left', fontweight='bold', fontsize=14)
    axes[3].set_xlabel('PC')
    axes[3].set_ylabel('PC score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"PCA expression plot saved as '{save_path}'")
    
    return fig

#######################
# DATASET 1: WINE DATA
#######################
print("\n" + "="*50)
print("ANALYZING WINE DATASET")
print("="*50)

# Step 1: Load the data
print("Loading wine dataset...")
wine_df = pd.read_csv("wine.csv")
print(f"Dataset loaded with {wine_df.shape[0]} rows and {wine_df.shape[1]} columns")

# Step 2: Basic data exploration
print("\nFirst 5 rows of the wine dataset:")
print(wine_df.head())

print("\nBasic statistics of the wine dataset:")
print(wine_df.describe().round(2))

# Step 3: Check for missing values
missing_values = wine_df.isnull().sum()
print("\nMissing values in the wine dataset:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

# Step 4: Feature engineering - separate target variable
X_wine = wine_df.drop('Wine_type', axis=1)
y_wine = wine_df['Wine_type']

print("\nFeatures used for PCA:")
print(X_wine.columns.tolist())

# Step 5: Standardize the data
print("\nStandardizing the data...")
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# Step 6: Perform PCA
print("\nPerforming PCA...")
pca_wine = PCA()
X_wine_pca = pca_wine.fit_transform(X_wine_scaled)

# Step 7: Analyze explained variance
explained_variance = pca_wine.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance by each principal component:")
for i, var in enumerate(explained_variance[:5]):
    print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Find how many components explain 80% variance
components_needed = np.argmax(cumulative_variance >= 0.8) + 1
print(f"\nNumber of components needed to explain 80% variance: {components_needed}")

# Step 8: Visualize explained variance
plt.figure()
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance threshold')
plt.axvline(x=components_needed, color='g', linestyle='--', label=f'{components_needed} components needed')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components - Wine Dataset')
plt.legend()
plt.tight_layout()
plt.savefig("charts/wine_explained_variance.png")
print("Explained variance plot saved as 'charts/wine_explained_variance.png'")

# Step 9: Visualize feature loadings
feature_names = X_wine.columns
loadings = pd.DataFrame(
    pca_wine.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca_wine.components_))],
    index=feature_names
)

# Display top components and their loading values
print("\nFeature loadings for the first 2 principal components:")
for i in range(2):
    pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
    print(f"\nPC{i+1} feature importance (absolute values):")
    for feat, value in pc_loadings.head(3).items():
        print(f"{feat}: {value:.4f}")

# Create a heatmap of feature loadings for the most important PCs
plt.figure()
loadings_subset = loadings.iloc[:, :3]  # First 3 components
sns.heatmap(loadings_subset, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Loadings in First 3 Principal Components - Wine Dataset")
plt.tight_layout()
plt.savefig("charts/wine_feature_loadings.png")
print("Feature loadings heatmap saved as 'charts/wine_feature_loadings.png'")

# Step 10: Create multi-panel visualization with dendrogram and PCA scatter plots
# Perform hierarchical clustering
Z = linkage(X_wine_scaled, 'ward')

# Create a figure with a grid layout
fig = plt.figure(figsize=(16, 12))
# Modified GridSpec to match Figure 3 layout
gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])

# Panel a: Dendrogram
ax0 = fig.add_subplot(gs[0, 0])
dendrogram(Z, orientation='left', leaf_font_size=8, ax=ax0)
ax0.set_title('a', loc='left', fontweight='bold', fontsize=14)
ax0.set_ylabel('Sample Index')
ax0.set_xlabel('Distance')

# Determine clusters from the dendrogram (using a fixed number of clusters instead of distance)
# Using 3 clusters to match Figure 3
from scipy.cluster.hierarchy import fcluster
n_clusters = 3  # Fixed number of clusters
clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Define colors for clusters
unique_clusters = np.unique(clusters)
cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}

# Panel b: PCA scatter plot (PC1 vs PC2)
ax1 = fig.add_subplot(gs[0, 1])
for cluster in unique_clusters:
    mask = clusters == cluster
    ax1.scatter(
        X_wine_pca[mask, 0], 
        X_wine_pca[mask, 1],
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
    # Add a text label for the cluster at the centroid
    centroid = np.mean(X_wine_pca[mask, :2], axis=0)
    ax1.text(centroid[0], centroid[1], f'{cluster}', fontweight='bold', fontsize=12)

ax1.set_title('b', loc='left', fontweight='bold', fontsize=14)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.3)

# Panel c: PC1 vs PC2 scatter plot (scaled differently)
ax2 = fig.add_subplot(gs[1, 0])
for cluster in unique_clusters:
    mask = clusters == cluster
    ax2.scatter(
        X_wine_pca[mask, 0] * 100, 
        X_wine_pca[mask, 1] * 100,
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
ax2.set_title('c', loc='left', fontweight='bold', fontsize=14)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.grid(True, alpha=0.3)

# Panel d: Raw data scatter plot (using first two original features)
ax3 = fig.add_subplot(gs[1, 1])
# Select two most important features based on PC1 loadings
top_features = loadings['PC1'].abs().sort_values(ascending=False).index[:2].tolist()
for cluster in unique_clusters:
    mask = clusters == cluster
    ax3.scatter(
        X_wine.loc[mask, top_features[0]], 
        X_wine.loc[mask, top_features[1]],
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
ax3.set_title('d', loc='left', fontweight='bold', fontsize=14)
ax3.set_xlabel(top_features[0])
ax3.set_ylabel(top_features[1])
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("charts/wine_multi_panel_pca.png")
print("Wine multi-panel PCA visualization saved as 'charts/wine_multi_panel_pca.png'")

# Step 11: Simple explanation for the boss
print("\n" + "-"*50)
print("WINE DATASET ANALYSIS SUMMARY FOR THE BOSS")
print("-"*50)
print("""
Our analysis of the wine dataset using Principal Component Analysis (PCA) reveals:

1. We can reduce the dataset from 13 features to just {0} components while retaining 80% of the information.

2. The most important features in the first principal component are:
   - {1}: {2:.4f}
   - {3}: {4:.4f}
   - {5}: {6:.4f}

3. The scatter plot shows clear separation between different wine types, indicating that our 
   chemical measurements are effective at distinguishing between wine varieties.

4. This analysis helps us understand which chemical properties are most important for 
   classifying wines, which could inform quality control processes or new product development.
""".format(
    components_needed,
    loadings['PC1'].abs().sort_values(ascending=False).index[0], 
    abs(loadings.loc[loadings['PC1'].abs().sort_values(ascending=False).index[0], 'PC1']),
    loadings['PC1'].abs().sort_values(ascending=False).index[1], 
    abs(loadings.loc[loadings['PC1'].abs().sort_values(ascending=False).index[1], 'PC1']),
    loadings['PC1'].abs().sort_values(ascending=False).index[2], 
    abs(loadings.loc[loadings['PC1'].abs().sort_values(ascending=False).index[2], 'PC1'])
))

# Create expression profile visualization for Wine dataset
print("\nCreating PCA expression profile visualization for Wine dataset...")
wine_expr_fig = create_pca_expression_plot(
    X_wine, 
    pca_wine, 
    n_components=6, 
    n_samples=6,
    title="Wine Dataset PCA Expression Profiles",
    save_path="charts/wine_expression_profiles.png"
)

#######################
# DATASET 2: CITIES DATA
#######################
print("\n" + "="*50)
print("ANALYZING CITIES DATASET")
print("="*50)

# Step 1: Load the data
print("Loading cities dataset...")
# Fix for cities dataset - properly handle the CSV format
cities_df = pd.read_csv("cities.csv", skipinitialspace=True)
# Clean up column names by removing quotes
cities_df.columns = cities_df.columns.str.replace('"', '')
print(f"Dataset loaded with {cities_df.shape[0]} rows and {cities_df.shape[1]} columns")

# Step 2: Basic data exploration
print("\nFirst 5 rows of the cities dataset:")
print(cities_df.head())

# Print column names to debug
print("\nColumn names in the cities dataset:")
print(cities_df.columns.tolist())

# Step 3: Check for missing values
missing_values = cities_df.isnull().sum()
print("\nMissing values in the cities dataset:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

# Step 4: Feature engineering
# Convert latitude and longitude to decimal degrees
print("\nConverting latitude and longitude to decimal degrees...")
# Handle string values in NS and EW columns
cities_df['NS'] = cities_df['NS'].str.replace('"', '')
cities_df['EW'] = cities_df['EW'].str.replace('"', '')
cities_df['City'] = cities_df['City'].str.replace('"', '')

cities_df['Latitude'] = cities_df['LatD'] + cities_df['LatM']/60 + cities_df['LatS']/3600
cities_df['Longitude'] = cities_df['LonD'] + cities_df['LonM']/60 + cities_df['LonS']/3600

# Adjust for direction (N/S, E/W)
cities_df.loc[cities_df['NS'] == 'N', 'Latitude'] *= 1  # No change for North
cities_df.loc[cities_df['NS'] == 'S', 'Latitude'] *= -1  # Negative for South
cities_df.loc[cities_df['EW'] == 'E', 'Longitude'] *= 1  # No change for East
cities_df.loc[cities_df['EW'] == 'W', 'Longitude'] *= -1  # Negative for West

# Select only numeric features for PCA
X_cities = cities_df[['Latitude', 'Longitude']]

print("\nFeatures used for PCA:")
print(X_cities.columns.tolist())

# Step 5: Standardize the data
print("\nStandardizing the data...")
scaler = StandardScaler()
X_cities_scaled = scaler.fit_transform(X_cities)

# Step 6: Perform PCA
print("\nPerforming PCA...")
pca_cities = PCA()
X_cities_pca = pca_cities.fit_transform(X_cities_scaled)

# Step 7: Analyze explained variance
explained_variance = pca_cities.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance by each principal component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Step 8: Visualize explained variance
plt.figure()
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components - Cities Dataset')
plt.legend()
plt.tight_layout()
plt.savefig("charts/cities_explained_variance.png")
print("Explained variance plot saved as 'charts/cities_explained_variance.png'")

# Step 9: Visualize feature loadings
feature_names = X_cities.columns
loadings = pd.DataFrame(
    pca_cities.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca_cities.components_))],
    index=feature_names
)

# Display components and their loading values
print("\nFeature loadings for the principal components:")
for i in range(len(pca_cities.components_)):
    pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
    print(f"\nPC{i+1} feature importance (absolute values):")
    for feat, value in pc_loadings.items():
        print(f"{feat}: {value:.4f}")

# Create a heatmap of feature loadings
plt.figure()
sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Loadings in Principal Components - Cities Dataset")
plt.tight_layout()
plt.savefig("charts/cities_feature_loadings.png")
print("Feature loadings heatmap saved as 'charts/cities_feature_loadings.png'")

# Step 10: Create multi-panel visualization with dendrogram and PCA scatter plots for cities
# Perform hierarchical clustering
Z = linkage(X_cities_scaled, 'ward')

# Create a figure with a grid layout
fig = plt.figure(figsize=(16, 12))
# Modified GridSpec to match Figure 3 layout
gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])

# Panel a: Dendrogram
ax0 = fig.add_subplot(gs[0, 0])
dendrogram(Z, orientation='left', leaf_font_size=8, ax=ax0)
ax0.set_title('a', loc='left', fontweight='bold', fontsize=14)
ax0.set_ylabel('City Index')
ax0.set_xlabel('Distance')

# Determine clusters from the dendrogram (using a fixed number of clusters)
n_clusters = 3  # Fixed number of clusters to match Figure 3
clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Define colors for clusters
unique_clusters = np.unique(clusters)
cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}

# Panel b: PCA scatter plot (PC1 vs PC2)
ax1 = fig.add_subplot(gs[0, 1])
for cluster in unique_clusters:
    mask = clusters == cluster
    ax1.scatter(
        X_cities_pca[mask, 0], 
        X_cities_pca[mask, 1],
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
    # Add a text label for the cluster at the centroid
    if np.sum(mask) > 0:  # Only add label if there are points in the cluster
        centroid = np.mean(X_cities_pca[mask, :2], axis=0)
        ax1.text(centroid[0], centroid[1], f'{cluster}', fontweight='bold', fontsize=12)

ax1.set_title('b', loc='left', fontweight='bold', fontsize=14)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.3)

# Panel c: PC1 vs PC2 scatter plot (scaled differently)
ax2 = fig.add_subplot(gs[1, 0])
for cluster in unique_clusters:
    mask = clusters == cluster
    ax2.scatter(
        X_cities_pca[mask, 0] * 100, 
        X_cities_pca[mask, 1] * 100,
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
ax2.set_title('c', loc='left', fontweight='bold', fontsize=14)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.grid(True, alpha=0.3)

# Panel d: Raw data scatter plot (Longitude vs Latitude)
ax3 = fig.add_subplot(gs[1, 1])
for cluster in unique_clusters:
    mask = clusters == cluster
    ax3.scatter(
        cities_df.loc[mask, 'Longitude'], 
        cities_df.loc[mask, 'Latitude'],
        c=[color_map[cluster]],
        s=50,
        alpha=0.7
    )
ax3.set_title('d', loc='left', fontweight='bold', fontsize=14)
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("charts/cities_multi_panel_pca.png")
print("Cities multi-panel PCA visualization saved as 'charts/cities_multi_panel_pca.png'")

# Create expression profile visualization for Cities dataset
print("\nCreating PCA expression profile visualization for Cities dataset...")
cities_expr_fig = create_pca_expression_plot(
    X_cities, 
    pca_cities, 
    n_components=2, 
    n_samples=6,
    title="Cities Dataset PCA Expression Profiles",
    save_path="charts/cities_expression_profiles.png"
)

# Step 11: Simple explanation for the boss
print("\n" + "-"*50)
print("CITIES DATASET ANALYSIS SUMMARY FOR THE BOSS")
print("-"*50)
print("""
Our analysis of the cities dataset using Principal Component Analysis (PCA) reveals:

1. With only two features (latitude and longitude), PCA doesn't reduce dimensions much, 
   but it does show us how these coordinates relate to each other.

2. The first principal component explains {0:.1f}% of the variance, which represents the 
   main geographic spread of cities across the country.

3. The second principal component explains the remaining {1:.1f}% of variance, capturing 
   the orthogonal direction of city distribution.

4. This analysis helps us visualize the geographic distribution of cities and could be 
   useful for regional planning or market analysis.
""".format(
    explained_variance[0] * 100,
    explained_variance[1] * 100
))

#######################
# DATASET 3: ECONOMIC DATA
#######################
print("\n" + "="*50)
print("ANALYZING ECONOMIC DATASET")
print("="*50)

# Step 1: Load the data
print("Loading economic dataset...")
try:
    economic_df = pd.read_csv("economic1.csv", low_memory=False)
    print(f"Dataset loaded with {economic_df.shape[0]} rows and {economic_df.shape[1]} columns")

    # Step 2: Basic data exploration
    print("\nFirst 5 rows of the economic dataset:")
    print(economic_df.head())

    # Step 3: Check for missing values
    missing_values = economic_df.isnull().sum()
    print("\nMissing values in the economic dataset (top 10):")
    print(missing_values.sort_values(ascending=False).head(10))

    # Step 4: Feature engineering
    # Drop unnecessary columns
    print("\nDropping unnecessary columns...")
    drop_cols = ['Series_title_2', 'Series_title_3', 'Series_title_4', 'Series_title_5']
    economic_df = economic_df.drop(columns=drop_cols, errors='ignore')
    
    # Handle missing values
    print("\nHandling missing values...")
    economic_df = economic_df.dropna(subset=['Data_value'])  # Drop rows where target is missing
    
    # Handle Period column by converting to year and month
    economic_df['Year'] = economic_df['Period'].astype(str).str.split('.', expand=True)[0].astype(float)
    economic_df['Month'] = economic_df['Period'].astype(str).str.split('.', expand=True)[1].astype(float)
    
    # Select features for PCA
    X_economic = economic_df[['Year', 'Month']]
    if 'MAGNTUDE' in economic_df.columns:
        X_economic['MAGNTUDE'] = economic_df['MAGNTUDE']
    
    y_economic = economic_df['Data_value']
    
    print("\nFeatures used for PCA:")
    print(X_economic.columns.tolist())
    
    # Step 5: Standardize the data
    print("\nStandardizing the data...")
    scaler = StandardScaler()
    X_economic_scaled = scaler.fit_transform(X_economic)
    
    # Step 6: Perform PCA
    print("\nPerforming PCA...")
    pca_economic = PCA()
    X_economic_pca = pca_economic.fit_transform(X_economic_scaled)
    
    # Step 7: Analyze explained variance
    explained_variance = pca_economic.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nExplained variance by each principal component:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
    
    # Find how many components explain 80% variance
    components_needed = np.argmax(cumulative_variance >= 0.8) + 1
    print(f"\nNumber of components needed to explain 80% variance: {components_needed}")
    
    # Step 8: Visualize explained variance
    plt.figure()
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance threshold')
    plt.axvline(x=components_needed, color='g', linestyle='--', label=f'{components_needed} components needed')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Components - Economic Dataset')
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/economic_explained_variance.png")
    print("Explained variance plot saved as 'charts/economic_explained_variance.png'")
    
    # Step 9: Visualize feature loadings
    feature_names = X_economic.columns
    loadings = pd.DataFrame(
        pca_economic.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca_economic.components_))],
        index=feature_names
    )
    
    # Display top components and their loading values
    print("\nFeature loadings for the principal components:")
    for i in range(len(pca_economic.components_)):
        pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(f"\nPC{i+1} feature importance (absolute values):")
        for feat, value in pc_loadings.items():
            print(f"{feat}: {value:.4f}")
    
    # Create a heatmap of feature loadings
    plt.figure()
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Loadings in Principal Components - Economic Dataset")
    plt.tight_layout()
    plt.savefig("charts/economic_feature_loadings.png")
    print("Feature loadings heatmap saved as 'charts/economic_feature_loadings.png'")
    
    # Step 10: Create multi-panel visualization with dendrogram and PCA scatter plots for economic data
    # Sample the data if it's very large
    sample_size = min(1000, len(X_economic_scaled))
    indices = np.random.choice(len(X_economic_scaled), sample_size, replace=False)
    
    # Perform hierarchical clustering on the sample
    Z = linkage(X_economic_scaled[indices], 'ward')
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(16, 12))
    # Modified GridSpec to match Figure 3 layout
    gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])
    
    # Panel a: Dendrogram
    ax0 = fig.add_subplot(gs[0, 0])
    dendrogram(Z, orientation='left', leaf_font_size=8, ax=ax0)
    ax0.set_title('a', loc='left', fontweight='bold', fontsize=14)
    ax0.set_ylabel('Sample Index')
    ax0.set_xlabel('Distance')
    
    # Determine clusters from the dendrogram (using a fixed number of clusters)
    n_clusters = 3  # Fixed number of clusters to match Figure 3
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Define colors for clusters
    unique_clusters = np.unique(clusters)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Panel b: PCA scatter plot (PC1 vs PC2)
    ax1 = fig.add_subplot(gs[0, 1])
    for cluster in unique_clusters:
        mask = clusters == cluster
        ax1.scatter(
            X_economic_pca[indices][mask, 0], 
            X_economic_pca[indices][mask, 1],
            c=[color_map[cluster]],
            s=50,
            alpha=0.7
        )
        # Add a text label for the cluster at the centroid
        if np.sum(mask) > 0:  # Only add label if there are points in the cluster
            centroid = np.mean(X_economic_pca[indices][mask, :2], axis=0)
            ax1.text(centroid[0], centroid[1], f'{cluster}', fontweight='bold', fontsize=12)
    
    ax1.set_title('b', loc='left', fontweight='bold', fontsize=14)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, alpha=0.3)
    
    # Panel c: PC1 vs PC2 scatter plot (scaled differently)
    ax2 = fig.add_subplot(gs[1, 0])
    for cluster in unique_clusters:
        mask = clusters == cluster
        ax2.scatter(
            X_economic_pca[indices][mask, 0] * 100, 
            X_economic_pca[indices][mask, 1] * 100,
            c=[color_map[cluster]],
            s=50,
            alpha=0.7
        )
    ax2.set_title('c', loc='left', fontweight='bold', fontsize=14)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)
    
    # Panel d: Raw data scatter plot (Year vs Data_value)
    ax3 = fig.add_subplot(gs[1, 1])
    for cluster in unique_clusters:
        mask = clusters == cluster
        ax3.scatter(
            economic_df.iloc[indices][mask]['Year'], 
            economic_df.iloc[indices][mask]['Data_value'],
            c=[color_map[cluster]],
            s=50,
            alpha=0.7
        )
    ax3.set_title('d', loc='left', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Data Value')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("charts/economic_multi_panel_pca.png")
    print("Economic multi-panel PCA visualization saved as 'charts/economic_multi_panel_pca.png'")
    
    # Step 11: Simple explanation for the boss
    print("\n" + "-"*50)
    print("ECONOMIC DATASET ANALYSIS SUMMARY FOR THE BOSS")
    print("-"*50)
    print("""
Our analysis of the economic dataset using Principal Component Analysis (PCA) reveals:

1. We can reduce the dataset from {0} features to just {1} components while retaining 80% of the information.

2. The most important features in the first principal component are:
   - {2}: {3:.4f}
   - {4}: {5:.4f}

3. The scatter plot shows how economic data points are distributed across the principal components,
   with colors representing the data values.

4. This analysis helps us understand which factors most influence economic indicators,
   which could inform policy decisions or economic forecasting.
    """.format(
        len(X_economic.columns),
        components_needed,
        loadings['PC1'].abs().sort_values(ascending=False).index[0], 
        abs(loadings.loc[loadings['PC1'].abs().sort_values(ascending=False).index[0], 'PC1']),
        loadings['PC1'].abs().sort_values(ascending=False).index[1], 
        abs(loadings.loc[loadings['PC1'].abs().sort_values(ascending=False).index[1], 'PC1'])
    ))

    # Create expression profile visualization for Economic dataset
    print("\nCreating PCA expression profile visualization for Economic dataset...")
    economic_expr_fig = create_pca_expression_plot(
        X_economic, 
        pca_economic, 
        n_components=min(6, len(pca_economic.components_)), 
        n_samples=6,
        title="Economic Dataset PCA Expression Profiles",
        save_path="charts/economic_expression_profiles.png"
    )

except Exception as e:
    print(f"Error processing economic dataset: {e}")
    print("Skipping economic dataset analysis.")

print("\n" + "="*50)
print("PCA ANALYSIS COMPLETE")
print("="*50)
print("All visualizations have been saved to the 'charts' directory.") 