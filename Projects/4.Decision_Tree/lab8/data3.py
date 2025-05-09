import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Select a subset of features for easier input
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points'
]
X = X[selected_features]

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train models with different strategies
models = {
    'Standard SVM': SVC(probability=True),
    'One-vs-Rest (OvA)': OneVsRestClassifier(SVC(probability=True)),
    'One-vs-One (OvO)': OneVsOneClassifier(SVC())
}

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nResults for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC curve and AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Plot Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, label=f'{name} (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()

plt.tight_layout()
plt.savefig('model_performance.png')
plt.close()

def get_new_data_point():
    """Get new data point from user with proper feature handling"""
    print("\nEnter new data for prediction:")
    print("\nFeature descriptions and value ranges:")
    print("Mean Radius: Average distance from center to points on perimeter (6-28)")
    print("Mean Texture: Standard deviation of gray-scale values (9-40)")
    print("Mean Perimeter: Mean size of the core tumor (43-190)")
    print("Mean Area: Mean area of the core tumor (143-2500)")
    print("Mean Smoothness: Local variation in radius lengths (0.05-0.16)")
    print("Mean Compactness: Perimeter² / area - 1.0 (0.02-0.35)")
    print("Mean Concavity: Severity of concave portions of the contour (0-0.43)")
    print("Mean Concave Points: Number of concave portions of the contour (0-0.20)")
    
    new_data = {}
    
    # Get Mean Radius
    while True:
        try:
            radius = float(input("\nEnter Mean Radius (6-28): "))
            if 6 <= radius <= 28:
                new_data['mean radius'] = radius
                break
            else:
                print("Please enter a value between 6 and 28.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Texture
    while True:
        try:
            texture = float(input("Enter Mean Texture (9-40): "))
            if 9 <= texture <= 40:
                new_data['mean texture'] = texture
                break
            else:
                print("Please enter a value between 9 and 40.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Perimeter
    while True:
        try:
            perimeter = float(input("Enter Mean Perimeter (43-190): "))
            if 43 <= perimeter <= 190:
                new_data['mean perimeter'] = perimeter
                break
            else:
                print("Please enter a value between 43 and 190.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Area
    while True:
        try:
            area = float(input("Enter Mean Area (143-2500): "))
            if 143 <= area <= 2500:
                new_data['mean area'] = area
                break
            else:
                print("Please enter a value between 143 and 2500.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Smoothness
    while True:
        try:
            smoothness = float(input("Enter Mean Smoothness (0.05-0.16): "))
            if 0.05 <= smoothness <= 0.16:
                new_data['mean smoothness'] = smoothness
                break
            else:
                print("Please enter a value between 0.05 and 0.16.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Compactness
    while True:
        try:
            compactness = float(input("Enter Mean Compactness (0.02-0.35): "))
            if 0.02 <= compactness <= 0.35:
                new_data['mean compactness'] = compactness
                break
            else:
                print("Please enter a value between 0.02 and 0.35.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Concavity
    while True:
        try:
            concavity = float(input("Enter Mean Concavity (0-0.43): "))
            if 0 <= concavity <= 0.43:
                new_data['mean concavity'] = concavity
                break
            else:
                print("Please enter a value between 0 and 0.43.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get Mean Concave Points
    while True:
        try:
            concave_points = float(input("Enter Mean Concave Points (0-0.20): "))
            if 0 <= concave_points <= 0.20:
                new_data['mean concave points'] = concave_points
                break
            else:
                print("Please enter a value between 0 and 0.20.")
        except ValueError:
            print("Please enter a valid number.")
    
    return pd.DataFrame([new_data])

# Example usage
while True:
    try:
        new_data = get_new_data_point()
        
        # Scale the new data
        new_data_scaled = scaler.transform(new_data)
        
        # Make predictions with all models
        for name, model in models.items():
            prediction = model.predict(new_data_scaled)[0]
            print(f"\n{name} Prediction:")
            print(f"Diagnosis: {'Malignant' if prediction == 1 else 'Benign'}")
            
            # Only show probabilities for models that support it
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(new_data_scaled)[0]
                print(f"Confidence: {max(probabilities) * 100:.2f}%")
        
        retry = input("\nDo you want to make another prediction? (y/n): ")
        if retry.lower() != 'y':
            break
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
        retry = input("Do you want to try again? (y/n): ")
        if retry.lower() != 'y':
            break
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        retry = input("Do you want to try again? (y/n): ")
        if retry.lower() != 'y':
            break

"""
Executive Summary: Breast Cancer Diagnosis Prediction Model
=======================================================

Dear Supervisor,

I've developed a machine learning model that predicts whether a breast cancer tumor is malignant or benign based on various medical measurements. This model demonstrates our capability to build accurate diagnostic tools that can assist medical professionals in making informed decisions.

Business Value:
-------------
1. Medical Diagnosis Support: The model uses eight key medical measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, and concave points) to predict tumor classification. This enables:
   - Quick preliminary diagnosis
   - Risk assessment
   - Treatment planning
   - Resource allocation optimization

2. Data-Driven Decision Making: The model provides both predictions and confidence levels, helping medical professionals make informed decisions about patient care and treatment strategies.

3. Efficiency in Analysis: By using carefully selected medical features, we demonstrate how to build effective predictive models that capture important diagnostic characteristics.

4. Multiple Analysis Approaches: We've implemented three different methods (Standard, One-vs-Rest, and One-vs-One) to show how we can optimize predictions for different diagnostic scenarios.

Practical Applications:
---------------------
1. Medical Diagnosis: Assist in preliminary tumor classification
2. Risk Assessment: Help identify high-risk cases
3. Resource Allocation: Optimize medical resource distribution
4. Treatment Planning: Support treatment decision-making

The model's success in predicting tumor classification with high confidence demonstrates our ability to build efficient, accurate predictive models that can support medical decision-making and improve patient care.

Best regards,
[Your Name]
""" 