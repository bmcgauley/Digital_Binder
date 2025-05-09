import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import requests
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Step 1: Loading and Preparing Data")
print("---------------------------------")

# Try to load the dataset
try:
    # First try using ucimlrepo
    credit_card_default = fetch_ucirepo(id=350)
    X = credit_card_default.data.features
    y = credit_card_default.data.targets
except Exception as e:
    print("Error loading from UCI repository, trying alternative source...")
    try:
        # Alternative method: Download directly from UCI website
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default_of_credit_card_clients.xls"
        response = requests.get(url)
        if response.status_code == 200:
            # Read the Excel file from memory
            X = pd.read_excel(io.BytesIO(response.content))
            y = X['Y']  # Target variable
            X = X.drop('Y', axis=1)  # Remove target from features
        else:
            raise Exception(f"Failed to download dataset. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error loading the dataset: {str(e)}")
        print("\nPlease try one of these solutions:")
        print("1. Check your internet connection")
        print("2. Try downloading the dataset manually from:")
        print("   https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default_of_credit_card_clients.xls")
        print("3. Make sure you have the required packages installed:")
        print("   pip install ucimlrepo requests openpyxl")
        exit(1)

# Display basic info
print(f"Dataset: Default of Credit Card Clients")
print(f"Total instances: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target distribution:")
print(y.value_counts())

# Check column names
print("\nFeature column names:")
print(X.columns.tolist())

# Select a reasonable subset of important features for easier analysis and interpretation
# Based on dataset documentation:
# X1: Amount of the given credit (NT dollar)
# X2: Gender (1=male, 2=female)
# X3: Education (1=graduate school, 2=university, 3=high school, 4=others)
# X4: Marital status (1=married, 2=single, 3=others)
# X5: Age (years)
# X6: Repayment status in Sept 2005 (-1=pay duly, 1=1 month delay, 2=2 months delay, etc.)
# X7: Repayment status in Aug 2005
# X8: Repayment status in July 2005
# X12: Bill amount in Sept 2005
# X13: Bill amount in Aug 2005
# X18: Payment amount in Sept 2005
# X19: Payment amount in Aug 2005
selected_features = [
    'X1',   # Credit limit
    'X2',   # Gender
    'X3',   # Education level
    'X4',   # Marital status
    'X5',   # Age
    'X6',   # Repayment status for Sept 2005
    'X7',   # Repayment status for Aug 2005
    'X8',   # Repayment status for July 2005
    'X12',  # Bill amount for Sept 2005
    'X13',  # Bill amount for Aug 2005
    'X18',  # Payment amount for Sept 2005
    'X19'   # Payment amount for Aug 2005
]

X = X[selected_features]

# Verify no missing values
print("\nMissing values in the dataset:")
print(X.isnull().sum())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStep 2: Data Visualization")
print("-------------------------")

# Plot 1: Distribution of the target variable
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.countplot(x=y.iloc[:, 0])
plt.title('Distribution of Credit Card Default')
plt.xlabel('Default (1 = Yes, 0 = No)')
plt.ylabel('Count')

# Plot 2: Age distribution by default status
plt.subplot(2, 2, 2)
sns.boxplot(x=y.iloc[:, 0], y=X['X5'])
plt.title('Age Distribution by Default Status')
plt.xlabel('Default (1 = Yes, 0 = No)')
plt.ylabel('Age')

# Plot 3: Credit limit distribution by default status
plt.subplot(2, 2, 3)
sns.boxplot(x=y.iloc[:, 0], y=X['X1'])
plt.title('Credit Limit Distribution by Default Status')
plt.xlabel('Default (1 = Yes, 0 = No)')
plt.ylabel('Credit Limit')

# Plot 4: Correlation heatmap
plt.subplot(2, 2, 4)
correlation = X.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('credit_card_distribution.png')
plt.close()

# Create PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.iloc[:, 0], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('PCA Visualization of Credit Card Default Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('credit_card_pca.png')
plt.close()

print("\nStep 3: Training Standard SVM")
print("-----------------------------")

# Create and train standard SVM
standard_svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=RANDOM_SEED
)

print("Training standard SVM...")
standard_svm.fit(X_train_scaled, y_train.values.ravel())

# Make predictions
y_pred_standard = standard_svm.predict(X_test_scaled)

# Calculate metrics
accuracy_standard = accuracy_score(y_test, y_pred_standard)
precision_standard = precision_score(y_test, y_pred_standard, average='weighted')
recall_standard = recall_score(y_test, y_pred_standard, average='weighted')
f1_standard = f1_score(y_test, y_pred_standard, average='weighted')

print("\nResults for Standard SVM:")
print(f"Accuracy: {accuracy_standard:.4f}")
print(f"Precision: {precision_standard:.4f}")
print(f"Recall: {recall_standard:.4f}")
print(f"F1-Score: {f1_standard:.4f}")

print("\nClassification Report for Standard SVM:")
print(classification_report(y_test, y_pred_standard))

# Create confusion matrix for standard SVM
cm_standard = confusion_matrix(y_test, y_pred_standard)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Standard SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_standard_svm.png')
plt.close()

print("\nStep 4: Training One-vs-Rest SVM")
print("--------------------------------")

# Create and train One-vs-Rest SVM
ova_svm = OneVsRestClassifier(
    SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_SEED
    )
)

print("Training One-vs-Rest SVM...")
ova_svm.fit(X_train_scaled, y_train.values.ravel())

# Make predictions
y_pred_ova = ova_svm.predict(X_test_scaled)

# Calculate metrics
accuracy_ova = accuracy_score(y_test, y_pred_ova)
precision_ova = precision_score(y_test, y_pred_ova, average='weighted')
recall_ova = recall_score(y_test, y_pred_ova, average='weighted')
f1_ova = f1_score(y_test, y_pred_ova, average='weighted')

print("\nResults for One-vs-Rest SVM:")
print(f"Accuracy: {accuracy_ova:.4f}")
print(f"Precision: {precision_ova:.4f}")
print(f"Recall: {recall_ova:.4f}")
print(f"F1-Score: {f1_ova:.4f}")

print("\nClassification Report for One-vs-Rest SVM:")
print(classification_report(y_test, y_pred_ova))

# Create confusion matrix for OVA SVM
cm_ova = confusion_matrix(y_test, y_pred_ova)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ova, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - One-vs-Rest (OvA)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_one-vs-rest_(ova).png')
plt.close()

print("\nStep 5: Training One-vs-One SVM")
print("--------------------------------")

# Create and train One-vs-One SVM
ovo_svm = OneVsOneClassifier(
    SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_SEED
    )
)

print("Training One-vs-One SVM...")
ovo_svm.fit(X_train_scaled, y_train.values.ravel())

# Make predictions
y_pred_ovo = ovo_svm.predict(X_test_scaled)

# Calculate metrics
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
precision_ovo = precision_score(y_test, y_pred_ovo, average='weighted')
recall_ovo = recall_score(y_test, y_pred_ovo, average='weighted')
f1_ovo = f1_score(y_test, y_pred_ovo, average='weighted')

print("\nResults for One-vs-One SVM:")
print(f"Accuracy: {accuracy_ovo:.4f}")
print(f"Precision: {precision_ovo:.4f}")
print(f"Recall: {recall_ovo:.4f}")
print(f"F1-Score: {f1_ovo:.4f}")

print("\nClassification Report for One-vs-One SVM:")
print(classification_report(y_test, y_pred_ovo))

# Create confusion matrix for OVO SVM
cm_ovo = confusion_matrix(y_test, y_pred_ovo)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ovo, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - One-vs-One (OvO)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_one-vs-one_(ovo).png')
plt.close()

# Create a comparison plot of model performances
models = ['Standard SVM', 'One-vs-Rest SVM', 'One-vs-One SVM']
accuracies = [accuracy_standard, accuracy_ova, accuracy_ovo]
precisions = [precision_standard, precision_ova, precision_ovo]
recalls = [recall_standard, recall_ova, recall_ovo]
f1_scores = [f1_standard, f1_ova, f1_ovo]

plt.figure(figsize=(12, 8))
x = np.arange(len(models))
width = 0.2

plt.bar(x - width*1.5, accuracies, width, label='Accuracy')
plt.bar(x - width/2, precisions, width, label='Precision')
plt.bar(x + width/2, recalls, width, label='Recall')
plt.bar(x + width*1.5, f1_scores, width, label='F1-Score')

plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

print("\nStep 6: Making Predictions")
print("-------------------------")

# Function to get user input
def get_user_input():
    print("\nEnter client information for credit default prediction:")
    
    # Get Credit Limit
    limit_bal = float(input("Enter credit limit (NT$): "))
    
    # Get Gender (1=male, 2=female)
    sex = int(input("Enter gender (1=male, 2=female): "))
    
    # Get Education Level
    education = int(input("Enter education level (1=graduate school, 2=university, 3=high school, 4=others): "))
    
    # Get Marital Status
    marriage = int(input("Enter marital status (1=married, 2=single, 3=others): "))
    
    # Get Age
    age = int(input("Enter age (years): "))
    
    # Get Previous Payment Status
    pay_1 = int(input("Enter repayment status for Sept 2005 (-1=paid duly, 1=1 month delay, 2=2 months delay, etc.): "))
    pay_2 = int(input("Enter repayment status for Aug 2005 (-1=paid duly, 1=1 month delay, 2=2 months delay, etc.): "))
    pay_3 = int(input("Enter repayment status for July 2005 (-1=paid duly, 1=1 month delay, 2=2 months delay, etc.): "))
    
    # Get Bill Amounts
    bill_amt1 = float(input("Enter bill amount for Sept 2005 (NT$): "))
    bill_amt2 = float(input("Enter bill amount for Aug 2005 (NT$): "))
    
    # Get Payment Amounts
    pay_amt1 = float(input("Enter payment amount for Sept 2005 (NT$): "))
    pay_amt2 = float(input("Enter payment amount for Aug 2005 (NT$): "))
    
    # Create DataFrame with user input
    user_data = pd.DataFrame({
        'X1': [limit_bal],
        'X2': [sex],
        'X3': [education],
        'X4': [marriage],
        'X5': [age],
        'X6': [pay_1],
        'X7': [pay_2],
        'X8': [pay_3],
        'X12': [bill_amt1],
        'X13': [bill_amt2],
        'X18': [pay_amt1],
        'X19': [pay_amt2]
    })
    
    return user_data

# Example usage
while True:
    try:
        # Get user input
        user_data = get_user_input()
        
        # Scale the user data
        user_data_scaled = scaler.transform(user_data)
        
        # Make predictions with all models
        print("\nPrediction Results:")
        
        # Standard SVM
        standard_pred = standard_svm.predict(user_data_scaled)[0]
        standard_prob = np.max(standard_svm.predict_proba(user_data_scaled)[0]) * 100
        print(f"\nStandard SVM Prediction:")
        print(f"Will {'DEFAULT' if standard_pred == 1 else 'NOT DEFAULT'} on credit card payment")
        print(f"Confidence: {standard_prob:.2f}%")
        
        # One-vs-Rest SVM
        ova_pred = ova_svm.predict(user_data_scaled)[0]
        ova_prob = np.max(ova_svm.predict_proba(user_data_scaled)[0]) * 100
        print(f"\nOne-vs-Rest SVM Prediction:")
        print(f"Will {'DEFAULT' if ova_pred == 1 else 'NOT DEFAULT'} on credit card payment")
        print(f"Confidence: {ova_prob:.2f}%")
        
        # One-vs-One SVM
        ovo_pred = ovo_svm.predict(user_data_scaled)[0]
        print(f"\nOne-vs-One SVM Prediction:")
        print(f"Will {'DEFAULT' if ovo_pred == 1 else 'NOT DEFAULT'} on credit card payment")
        
        # Ask if user wants to make another prediction
        another = input("\nMake another prediction? (y/n): ")
        if another.lower() != 'y':
            break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        retry = input("Do you want to try again? (y/n): ")
        if retry.lower() != 'y':
            break

print("\nCredit Card Default Prediction complete!")
