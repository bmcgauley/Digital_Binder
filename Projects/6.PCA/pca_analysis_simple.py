import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the economic data
print("Loading economic1.csv...")
df = pd.read_csv("economic1.csv", low_memory=False)  # Added low_memory back to handle mixed dtypes
print("Data loaded successfully!")

# Display basic information
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

# Identify the target variable
print(f"\nTarget variable 'Data_value' statistics:")
print(f"Mean: {df['Data_value'].mean():.2f}")
print(f"Min: {df['Data_value'].min():.2f}")
print(f"Max: {df['Data_value'].max():.2f}")

# Handle missing values - show counts first
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Instead of dropping all rows with NaN, just drop rows where target is NaN
# and drop columns that have too many missing values
df_clean = df.copy()
# Drop rows where target variable is missing
df_clean = df_clean.dropna(subset=['Data_value'])
print(f"\nRows after dropping target nulls: {len(df_clean)} (removed {len(df) - len(df_clean)} rows)")

# Drop columns with high percentage of missing values
# or columns we won't use for analysis
drop_cols = []
for col in df_clean.columns:
    # If more than 50% values are missing, drop the column
    if df_clean[col].isnull().mean() > 0.5:
        drop_cols.append(col)
        
# Also drop Series_title columns as they're likely redundant with other info
title_cols = [col for col in df_clean.columns if 'Series_title_' in col]
drop_cols.extend(title_cols)
drop_cols = list(set(drop_cols))  # Remove duplicates

print(f"\nDropping {len(drop_cols)} columns with too many missing values:")
print(drop_cols)
df_clean = df_clean.drop(columns=drop_cols)

print(f"\nClean dataset shape: {df_clean.shape}")

# Handle Period column by converting to year and month
df_clean['Year'] = df_clean['Period'].astype(str).str.split('.', expand=True)[0].astype(float)
df_clean['Month'] = df_clean['Period'].astype(str).str.split('.', expand=True)[1].astype(float)

# Identify numeric and categorical columns
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Remove the target variable from features
if 'Data_value' in numeric_cols:
    numeric_cols.remove('Data_value')
print(f"\nNumeric columns found: {len(numeric_cols)}")
print(numeric_cols)

categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {len(categorical_cols)}")
print(categorical_cols)

# Sample the data if it's very large (for faster processing)
sample_size = min(50000, len(df_clean))
df_sample = df_clean.sample(sample_size, random_state=42)

# Create a correlation matrix of numeric features
print("\nCreating correlation matrix of numeric features...")
plt.figure(figsize=(12, 10))
numeric_for_corr = numeric_cols.copy()
# If there are too many numeric columns, select a subset
if len(numeric_for_corr) > 10:
    numeric_for_corr = ['Year', 'Month']
    if 'MAGNTUDE' in df_clean.columns:
        numeric_for_corr.append('MAGNTUDE')
# Add 'Data_value' for correlation analysis
numeric_for_corr.append('Data_value')

corr_matrix = df_sample[numeric_for_corr].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix_numeric.png")
print("Correlation matrix saved as 'correlation_matrix_numeric.png'")

# Prepare feature set for PCA
print("\nPreparing final feature set for PCA...")
# Start with important numeric features
X_features = ['Year', 'Month']
if 'MAGNTUDE' in df_clean.columns:
    X_features.append('MAGNTUDE')

# Add encoded categorical features (limiting to key ones to avoid too many features)
key_categorical = []
if 'Subject' in df_clean.columns:
    key_categorical.append('Subject')
if 'Group' in df_clean.columns:
    key_categorical.append('Group')
if 'Series_reference' in df_clean.columns:
    key_categorical.append('Series_reference')

# Create the feature dataframe
X = df_sample[X_features].copy()

# Add encoded categorical columns (for the most important ones)
print("\nEncoding categorical features...")
for cat_col in key_categorical[:2]:  # Limit to top 2 categorical columns to avoid feature explosion
    col_name = f"{cat_col}_Code"
    X[col_name] = df_sample[cat_col].astype('category').cat.codes
    print(f"Encoded '{cat_col}' as '{col_name}'")

# Fill any remaining missing values
print("\nFilling any remaining missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print("\nFinal features for PCA:")
for i, col in enumerate(X.columns):
    print(f"{i+1}. {col}")
print(f"\nFeature dataframe shape: {X.shape}")

# Create correlation matrix of final features
print("\nCreating correlation matrix of final features...")
plt.figure(figsize=(12, 10))
# Add target for correlation analysis
X_with_target = X.copy()
X_with_target['Data_value'] = df_sample['Data_value'].values
corr_matrix = X_with_target.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix (including target)")
plt.tight_layout()
plt.savefig("correlation_matrix_features.png")
print("Feature correlation matrix saved as 'correlation_matrix_features.png'")

# Scale the data
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
print("\nPerforming PCA...")
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

# Plot explained variance
print("\nCreating explained variance plot...")
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, 
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', 
         label='Cumulative explained variance')

# Add threshold lines
main_threshold = 0.8  # We'll highlight the 80% threshold
plt.axhline(y=main_threshold, color='r', linestyle='--', 
           label=f'{main_threshold*100:.0f}% variance threshold')
components_for_main = np.argmax(cumulative_variance >= main_threshold) + 1
plt.axvline(x=components_for_main, color='g', linestyle='--', 
           label=f'{components_for_main} components needed')

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("pca_explained_variance.png")
print("Explained variance plot saved as 'pca_explained_variance.png'")

# Create a clearer feature importance visualization
print("\nAnalyzing feature importance in principal components...")
feature_names = X.columns
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=feature_names
)

# Display top components and their loading values
n_display = min(3, len(pca.components_))
print("\nFeature loadings for the first few principal components:")
for i in range(n_display):
    pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
    print(f"\nPC{i+1} feature importance (absolute values):")
    for feat, value in pc_loadings.items():
        print(f"{feat}: {value:.4f}")

# Create a heatmap of feature loadings for the most important PCs
n_components_to_plot = min(4, len(pca.components_))
plt.figure(figsize=(12, 8))
loadings_subset = loadings.iloc[:, :n_components_to_plot]
sns.heatmap(loadings_subset, annot=True, cmap='coolwarm', fmt='.2f')
plt.title(f"Feature Loadings in First {n_components_to_plot} Principal Components")
plt.tight_layout()
plt.savefig("feature_loadings.png")
print("Feature loadings heatmap saved as 'feature_loadings.png'")

# Create a more detailed bar plot for feature importance in PC1 and PC2
plt.figure(figsize=(12, 6))
# Plot PC1 feature importance
plt.subplot(1, 2, 1)
pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
colors = ['#1f77b4' if x >= 0 else '#d62728' for x in loadings.loc[pc1_loadings.index, 'PC1']]
pc1_loadings.plot(kind='bar', color=colors)
plt.title('PC1 Feature Importance')
plt.ylabel('Absolute Loading Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot PC2 feature importance
plt.subplot(1, 2, 2)
pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
colors = ['#1f77b4' if x >= 0 else '#d62728' for x in loadings.loc[pc2_loadings.index, 'PC2']]
pc2_loadings.plot(kind='bar', color=colors)
plt.title('PC2 Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("pc_feature_importance.png")
print("PC feature importance plots saved as 'pc_feature_importance.png'")

# Create 2D PCA scatter plot
print("\nCreating 2D PCA scatter plot...")
plt.figure(figsize=(10, 8))

# Color by target variable (data value)
target_values = df_sample['Data_value'].values
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_values, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Data Value (target)')

# Add some annotations to show the directions of original features
# This helps interpret what the PCs mean
if len(pca.components_) >= 2:
    n_features = len(feature_names)
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

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA Projection (colored by target value)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_2d_scatter.png")
print("2D PCA scatter plot saved as 'pca_2d_scatter.png'")

print("\nPCA analysis complete! Check the generated PNG files for visualizations.")

# Print a summary of the most important features
print("\n=== PCA ANALYSIS SUMMARY ===")
print(f"Total number of components needed for 80% variance: {components_for_main}")
print("\nMost important features (based on PC1 and PC2):")
for feat, value in pc1_loadings.head(3).items():
    print(f"PC1 - {feat}: {loadings.loc[feat, 'PC1']:.4f}")
for feat, value in pc2_loadings.head(3).items():
    print(f"PC2 - {feat}: {loadings.loc[feat, 'PC2']:.4f}")
print("\nThese features contribute most to the variance in your dataset.")
print("For your economic data, these are the key variables you should focus on.") 