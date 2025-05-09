import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# Step 1: Load the dataset
print("\nStep 1: Loading dataset...")
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print("\nInitial dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Step 2: Explore the dataset
print("\nStep 2: Exploring the dataset...")
print("\nDataset sample:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDataset statistics:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Step 3: Feature Engineering and Preprocessing
print("\nStep 3: Feature Engineering and Preprocessing...")

# Separate features and target
X = df.drop(['target', 'diagnosis'], axis=1)
y = df['target']

print("\nFeature set shape:", X.shape)
print("Target set shape:", y.shape)

# Apply StandardScaler to the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
print("\nStep 4: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Create and train the SVM model
print("\nStep 5: Creating and training the SVM model...")
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train, y_train)
print("Model training complete")

# Step 6: Evaluate the model
print("\nStep 6: Evaluating the model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Generate the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Step 7: Visualize the results
print("\nStep 7: Visualizing the results...")

# Apply PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for easier plotting
pca_df = pd.DataFrame(
    data=X_pca, 
    columns=['Principal Component 1', 'Principal Component 2']
)
pca_df['Target'] = y.values
pca_df['Diagnosis'] = pca_df['Target'].map({0: 'Malignant', 1: 'Benign'})

# Plot the PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2',
    hue='Diagnosis',
    palette='viridis',
    data=pca_df,
    alpha=0.7
)

plt.title('PCA of Breast Cancer Features')
plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
plt.legend(title='Diagnosis')
plt.tight_layout()
plt.savefig('breast_cancer_pca_plot.png')
print("PCA plot saved as 'breast_cancer_pca_plot.png'")

# Visualize feature importance by training a linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Get feature importance from the linear SVM
importance = np.abs(linear_svm.coef_[0])
feature_names = X.columns

# Create feature importance DataFrame
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance in Linear SVM')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# Visualize the correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = df.drop('diagnosis', axis=1).corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Correlation matrix plot saved as 'correlation_matrix.png'")

# Step 8: Create a function for predicting new data points
print("\nStep 8: Creating a function for predicting new data points...")

def get_new_data_point():
    """Prompts the user to input a new data point and returns it as a list."""
    
    print("\nEnter new data for prediction:")
    
    new_data = {}
    
    for col in X.columns:
        while True:
            try:
                value = float(input(f"Enter value for {col}: "))
                new_data[col] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return pd.DataFrame([new_data])

# Example of prediction (commented out to avoid immediate execution)
"""
print("\nEnter data for a new breast cancer case:")
new_data = get_new_data_point()

if new_data is not None:
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)[0]
    probabilities = model.predict_proba(new_data_scaled)[0]
    
    print("\nPrediction result:", "Malignant" if prediction == 0 else "Benign")
    print("Prediction probabilities:", probabilities)
"""

print("\nSVM model implementation complete!")