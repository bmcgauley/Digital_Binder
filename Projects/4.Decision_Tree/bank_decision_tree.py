#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IS170 Lab Decision Tree
Bank Marketing Dataset Analysis using Decision Tree

This script implements a Decision Tree model on the bank marketing dataset from UCI ML Repository.
The model predicts whether a client will subscribe to a term deposit (variable y).
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import sys
import subprocess

# Optional imports for visualization
try:
    from sklearn.tree import export_graphviz
    import pydotplus
    from IPython.display import Image
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Visualization libraries not available. Tree visualization will be skipped.")
    VISUALIZATION_AVAILABLE = False

# Check if Graphviz is installed and accessible
def is_graphviz_installed():
    try:
        # Try to run dot -V to check if Graphviz is installed
        subprocess.run(['dot', '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Print explanation for the audience and boss
print("=" * 80)
print("Bank Marketing Dataset Analysis with Decision Tree")
print("=" * 80)
print("""
This analysis uses Decision Tree algorithm to predict whether a client will subscribe to a term deposit 
based on their demographic and banking information. The model can help marketing teams identify 
potential customers who are more likely to subscribe to term deposits, allowing for more targeted 
and efficient marketing campaigns.
""")
print("=" * 80)

# Load the dataset
print("Loading and preparing the dataset...")
df = pd.read_csv("bank.csv", sep=";")

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")  # -1 for the target variable
print("\nFirst few rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics for Numerical Features:")
print(df.describe())

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Preprocess the data
print("\nPreprocessing the data...")

# Encode categorical variables
# Create a copy of the dataframe to avoid modifying the original data
df_encoded = df.copy()

# Initialize a label encoder for categorical features
label_encoder = LabelEncoder()

# List of categorical columns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

# Encode each categorical column
for col in categorical_cols:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

print("\nAfter encoding categorical variables:")
print(df_encoded.head())

# Split dataset into features and target variable
X = df_encoded.drop('y', axis=1)  # Features (all columns except 'y')
y = df_encoded['y']  # Target variable

# Split dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create and train the Decision Tree classifier
print("\nTraining the Decision Tree classifier...")
# Using a moderate depth to show more decision paths but avoid overfitting
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")

# Display detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualize feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance)

# Create a bar chart of feature importance
try:
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Decision Tree Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance chart saved as 'feature_importance.png'")
except Exception as e:
    print(f"Could not create feature importance chart due to error: {e}")

# Visualize the decision tree - this is the main focus as requested
print("\n" + "=" * 80)
print("DECISION TREE VISUALIZATION")
print("=" * 80)

if VISUALIZATION_AVAILABLE:
    have_graphviz = is_graphviz_installed()
    if not have_graphviz:
        print("\nWARNING: Graphviz software is not installed or not in PATH.")
        print("To install Graphviz on Windows:")
        print("1. Download from https://www.graphviz.org/download/")
        print("2. Add the bin directory to your PATH environment variable")
        print("\nAttempting to create visualization anyway...")
    
    try:
        # Create tree visualization with full depth to show all decisions
        dot_data = export_graphviz(
            dt_model,
            out_file=None,
            feature_names=X.columns,
            class_names=['No', 'Yes'],
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=5  # Show full tree up to max_depth
        )
        
        graph = pydotplus.graph_from_dot_data(dot_data)
        
        # Use different colors for different classes
        for i, node in enumerate(graph.get_node_list()):
            if node.get_name() not in ('node', 'edge'):
                if 'samples = ' in node.get_attributes().get('label', ''):
                    # Check if this is a leaf node
                    if 'value = [' in node.get_attributes().get('label', ''):
                        values = node.get_attributes()['label']
                        if 'class = Yes' in values:
                            node.set_fillcolor('#aaffaa')  # Light green for "Yes" nodes
                        else:
                            node.set_fillcolor('#ffaaaa')  # Light red for "No" nodes
        
        # Save the visualization with higher resolution
        graph.write_png('decision_tree.png')
        print("Complete decision tree visualization saved as 'decision_tree.png'")
        print("Please open this file to see all the decision paths in the model.")
        
        # For a more detailed tree, create a text representation
        with open('decision_tree_text.txt', 'w') as f:
            # Redirect stdout to the file
            old_stdout = sys.stdout
            sys.stdout = f
            
            # Print tree rules
            from sklearn.tree import _tree
            
            def tree_to_text(tree, feature_names):
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                ]
                
                def recurse(node, depth):
                    indent = "  " * depth
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        print(f"{indent}if {name} <= {threshold:.4f}:")
                        recurse(tree_.children_left[node], depth + 1)
                        print(f"{indent}else:  # if {name} > {threshold:.4f}")
                        recurse(tree_.children_right[node], depth + 1)
                    else:
                        class_prob = tree_.value[node][0]
                        total = sum(class_prob)
                        prob_no = class_prob[0] / total
                        prob_yes = class_prob[1] / total
                        predicted_class = "No" if prob_no > prob_yes else "Yes"
                        print(f"{indent}class: {predicted_class} (probability: {max(prob_no, prob_yes):.4f})")
                
                print("DECISION TREE RULES:")
                recurse(0, 0)
            
            tree_to_text(dt_model, X.columns)
            
            # Restore stdout
            sys.stdout = old_stdout
        
        print("Detailed text representation of decision rules saved as 'decision_tree_text.txt'")
            
    except Exception as e:
        print(f"Could not create decision tree visualization due to error: {e}")
        print("Make sure you have 'graphviz' software installed on your system.")
else:
    print("\nCannot create decision tree visualization - required libraries are not available.")
    print("Please install graphviz and pydotplus packages.")

# Function to predict for new data points entered by the user
def predict_new_customer():
    print("\n" + "=" * 80)
    print("New Customer Prediction")
    print("=" * 80)
    print("Enter the following details to predict if a customer will subscribe to a term deposit:")
    
    try:
        # Get user inputs for numerical features
        age = int(input("Age: "))
        balance = float(input("Account balance (in euros): "))
        day = int(input("Day of month for last contact (1-31): "))
        duration = int(input("Duration of last contact (in seconds): "))
        campaign = int(input("Number of contacts during this campaign: "))
        pdays = int(input("Days since the client was last contacted (-1 means never contacted): "))
        previous = int(input("Number of contacts before this campaign: "))
        
        # Get user inputs for categorical features with options
        print("\nOptions for 'job': admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown")
        job = input("Job type: ").lower()
        
        print("\nOptions for 'marital': divorced, married, single, unknown")
        marital = input("Marital status: ").lower()
        
        print("\nOptions for 'education': primary, secondary, tertiary, unknown")
        education = input("Education level: ").lower()
        
        print("\nOptions for 'default': yes, no, unknown")
        default = input("Has credit in default? ").lower()
        
        print("\nOptions for 'housing': yes, no, unknown")
        housing = input("Has housing loan? ").lower()
        
        print("\nOptions for 'loan': yes, no, unknown")
        loan = input("Has personal loan? ").lower()
        
        print("\nOptions for 'contact': cellular, telephone, unknown")
        contact = input("Contact communication type: ").lower()
        
        print("\nOptions for 'month': jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec")
        month = input("Month of last contact: ").lower()
        
        print("\nOptions for 'poutcome': failure, other, success, unknown")
        poutcome = input("Outcome of the previous marketing campaign: ").lower()
        
        # Create a dictionary for the new customer data
        new_customer = {
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'day': [day],
            'month': [month],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome]
        }
        
        # Convert to DataFrame
        new_df = pd.DataFrame(new_customer)
        
        # Encode categorical variables to match the training data
        for col in categorical_cols[:-1]:  # Excluding 'y' which is the target
            # Get unique values from training data for this column
            unique_values = df[col].unique()
            
            # Check if the provided value is valid
            if new_df[col][0] not in unique_values:
                print(f"Warning: Provided value '{new_df[col][0]}' for '{col}' is not in the training data.")
                print(f"Available options are: {', '.join(unique_values)}")
                # Assigning a default value
                new_df[col][0] = unique_values[0]
                print(f"Using '{unique_values[0]}' as default.")
            
            # Fit and transform
            new_df[col] = label_encoder.fit_transform(df[col].append(new_df[col])).tail(1).values
        
        # Make prediction
        prediction = dt_model.predict(new_df)
        probability = dt_model.predict_proba(new_df)
        
        print("\n" + "=" * 80)
        print("Prediction Result:")
        if prediction[0] == 1:
            print(f"The customer is LIKELY to subscribe to a term deposit (Confidence: {probability[0][1]:.2f})")
        else:
            print(f"The customer is UNLIKELY to subscribe to a term deposit (Confidence: {probability[0][0]:.2f})")
        print("=" * 80)
        
        # Show the decision path for this customer
        print("\nDecision path for this customer:")
        node_indicator = dt_model.decision_path(new_df)
        leaf_id = dt_model.apply(new_df)
        
        feature = dt_model.tree_.feature
        threshold = dt_model.tree_.threshold
        
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        for node_id in node_index:
            # Continue to the next node if it's a leaf
            if leaf_id[0] == node_id:
                print(f"🏁 Final prediction: {'Yes' if prediction[0] == 1 else 'No'}")
                continue
                
            # Check if the sample goes through the left or right branch
            if new_df.iloc[0, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
                direction = "left"
            else:
                threshold_sign = ">"
                direction = "right"
                
            print(f"➡️ {X.columns[feature[node_id]]} {threshold_sign} {threshold[node_id]:.4f} ({direction} branch)")
        
    except Exception as e:
        print(f"Error processing input: {e}")
        print("Please make sure you enter valid values for all fields.")

# Ask if user wants to make predictions for new customers
print("\n" + "=" * 80)
print("Would you like to predict for a new customer? This will show you the exact")
print("decision path taken for that customer through the decision tree.")
print("=" * 80)

while True:
    response = input("\nPredict for a new customer? (yes/no): ").lower()
    if response == 'yes':
        predict_new_customer()
    else:
        print("\nThank you for using the Bank Marketing Decision Tree Predictor!")
        break 