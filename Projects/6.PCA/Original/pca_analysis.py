import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.gridspec import GridSpec
import os

class PCAAnalyzer:
    def __init__(self, data, target_column=None):
        """
        Initialize PCA Analyzer
        
        Parameters:
        -----------
        data : pandas DataFrame
            The input dataset
        target_column : str, optional
            Name of the target column if exists
        """
        self.data = data
        self.target_column = target_column
        self.features = None
        self.pca = None
        self.scaler = None
        self.scaled_features = None
        self.explained_variance_ratio = None
        
        # Create charts directory if it doesn't exist
        if not os.path.exists('charts'):
            os.makedirs('charts')
            
        # Set up the plots to be more readable
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def explore_data(self):
        """Print basic information about the dataset"""
        print(f"\nDataset Overview:")
        print(f"Shape: {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nDescriptive Statistics:")
        print(self.data.describe().round(2))
        print("\nMissing Values:")
        print(self.data.isnull().sum())

    def prepare_features(self):
        """Prepare features for PCA analysis"""
        if self.target_column and self.target_column in self.data.columns:
            self.features = self.data.drop(columns=[self.target_column])
        else:
            self.features = self.data.copy()
            
        # Remove any non-numeric columns
        self.features = self.features.select_dtypes(include=['float64', 'int64'])
        
        return self.features.columns.tolist()

    def standardize_features(self):
        """Standardize the features"""
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        return self.scaled_features

    def perform_pca(self, n_components=None):
        """Perform PCA analysis"""
        if self.scaled_features is None:
            self.standardize_features()
            
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(self.scaled_features)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return transformed_features

    def plot_explained_variance(self, threshold=0.6):
        """
        Plot the explained variance ratio
        
        Parameters:
        -----------
        threshold : float
            Threshold for cumulative explained variance
        """
        cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        
        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Plot cumulative explained variance
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                cumulative_variance_ratio, 
                'b-', 
                label='Cumulative explained variance')
        
        # Plot individual explained variance
        plt.bar(range(1, len(self.explained_variance_ratio) + 1),
                self.explained_variance_ratio,
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
        plt.title('Explained Variance by Components')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save the plot
        plt.savefig('charts/explained_variance.png')
        plt.close()
        
        return n_components_needed

    def get_top_components(self, n_components=3, n_top_features=5):
        """
        Get the top features for each principal component
        
        Parameters:
        -----------
        n_components : int
            Number of components to analyze
        n_top_features : int
            Number of top features to show per component
        """
        feature_names = self.features.columns
        component_details = []
        
        for i in range(n_components):
            component = self.pca.components_[i]
            top_indices = np.abs(component).argsort()[-n_top_features:][::-1]
            
            component_info = {
                f"PC{i+1}": {
                    "explained_variance": self.explained_variance_ratio[i],
                    "top_features": [
                        (feature_names[idx], component[idx])
                        for idx in top_indices
                    ]
                }
            }
            component_details.append(component_info)
            
        return component_details

    def generate_report(self, n_components_needed):
        """Generate a markdown report of the PCA analysis"""
        report = f"""
# PCA Analysis Report

## Dataset Overview
- Number of samples: {self.data.shape[0]}
- Number of features: {self.features.shape[1]}
- Missing values: {self.data.isnull().sum().sum()}

## PCA Results
- Number of components needed to explain {60}% of variance: {n_components_needed}
- Total variance explained by first 3 components: {sum(self.explained_variance_ratio[:3]):.2%}

## Component Breakdown
"""
        
        top_components = self.get_top_components(3)
        for component in top_components:
            for pc_name, details in component.items():
                report += f"\n### {pc_name}\n"
                report += f"- Explained variance: {details['explained_variance']:.2%}\n"
                report += "- Top contributing features:\n"
                for feature, weight in details['top_features']:
                    report += f"  * {feature}: {weight:.3f}\n"
        
        # Save the report
        with open('pca_analysis_report.md', 'w') as f:
            f.write(report)
        
        return report

def main():
    """Main function to run the PCA analysis"""
    # Load the data
    try:
        wine_df = pd.read_csv("wine.csv")
    except FileNotFoundError:
        print("Please ensure the wine.csv file is in the current directory")
        return

    # Initialize PCA Analyzer
    pca_analyzer = PCAAnalyzer(wine_df, target_column='class')
    
    # Explore the data
    pca_analyzer.explore_data()
    
    # Prepare and standardize features
    pca_analyzer.prepare_features()
    pca_analyzer.standardize_features()
    
    # Perform PCA
    pca_analyzer.perform_pca()
    
    # Plot explained variance
    n_components_needed = pca_analyzer.plot_explained_variance(threshold=0.6)
    
    # Generate and save report
    report = pca_analyzer.generate_report(n_components_needed)
    print("\nAnalysis complete! Check the generated report and charts in the output directory.")

if __name__ == "__main__":
    main() 