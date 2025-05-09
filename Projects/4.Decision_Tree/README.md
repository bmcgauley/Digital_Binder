# Bank Marketing Decision Tree Analysis

## Overview
This project implements a Decision Tree classification model to predict whether a client will subscribe to a term deposit based on the UCI Bank Marketing dataset. The model analyzes various client attributes such as age, job type, marital status, and banking history to make predictions.

## Dataset
The dataset used is `bank.csv`, which contains information about bank clients and their response to a previous marketing campaign. The target variable (`y`) indicates whether the client subscribed to a term deposit (yes/no).

### Features:
- **age**: client age (numeric)
- **job**: type of job (categorical)
- **marital**: marital status (categorical)
- **education**: education level (categorical)
- **default**: has credit in default? (categorical)
- **balance**: average yearly balance in euros (numeric)
- **housing**: has housing loan? (categorical)
- **loan**: has personal loan? (categorical)
- **contact**: contact communication type (categorical)
- **day**: day of month of last contact (numeric)
- **month**: month of last contact (categorical)
- **duration**: duration of last contact in seconds (numeric)
- **campaign**: number of contacts during this campaign (numeric)
- **pdays**: days since client was last contacted (-1 means never contacted) (numeric)
- **previous**: number of contacts before this campaign (numeric)
- **poutcome**: outcome of the previous marketing campaign (categorical)
- **y**: has the client subscribed to a term deposit? (target variable)

## Requirements
To run this script, you need the following Python packages:
- pandas
- numpy
- matplotlib
- scikit-learn
- pydotplus (for tree visualization)
- graphviz (for tree visualization)

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn pydotplus graphviz
```

Additionally, you need to have Graphviz installed on your system to generate the decision tree visualization:
- **Windows**: Download and install from [Graphviz website](https://www.graphviz.org/download/)
- **Mac**: `brew install graphviz`
- **Linux**: `apt-get install graphviz`

## How to Run
1. Make sure the `bank.csv` file is in the same directory as the script.
2. Run the script:
   ```bash
   python bank_decision_tree.py
   ```
3. The script will load the data, train the model, and display various metrics and visualizations.
4. You can enter information for new customers to get predictions when prompted.

## Outputs
The script generates:

1. **Console Output**:
   - Dataset summary and statistics
   - Model evaluation metrics (accuracy, precision, recall, F1-score)
   - Classification report and confusion matrix
   - Feature importance ranking

2. **Visualization Files**:
   - `feature_importance.png`: Bar chart showing the importance of each feature
   - `decision_tree.png`: Visual representation of the decision tree (limited to 3 levels for readability)

3. **Interactive Predictions**:
   - The script allows you to input information for new customers and predicts whether they will subscribe to a term deposit

## Model Explanation
The Decision Tree algorithm creates a tree-like model of decisions based on feature values. Each internal node represents a "test" on a feature, each branch represents the outcome of the test, and each leaf node represents a class label.

In this implementation:
- The tree is limited to a maximum depth of 5 to prevent overfitting
- The model is evaluated using standard classification metrics
- Feature importance is calculated to understand which factors most strongly influence the prediction

## Note on the Decision Tree Node
In the decision tree visualization, each node contains important information:
- The feature and threshold used for splitting
- The Gini impurity measure
- Sample counts for each class
- The predicted class

Understanding these nodes helps interpret how the model makes decisions and which features have the most discriminative power at different levels of the tree.

## Author
IS170 Lab Student 