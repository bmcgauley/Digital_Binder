import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the Loan Default Dataset
print("Loading Loan Default Dataset...")

# Load the dataset
try:
    df = pd.read_csv('Loan_Default.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print("Error: File 'Loan_Default.csv' not found.")
    exit()

# Print initial class distribution
print("\nInitial class distribution:")
class_dist = df['Status'].value_counts(normalize=True) * 100
print(class_dist)

# Select features
base_numerical_features = ['loan_amount', 'term', 'income', 'Credit_Score', 'dtir1']
base_categorical_features = ['Gender', 'loan_type', 'loan_purpose', 'Credit_Worthiness', 'occupancy_type']

# Keep only needed features
df = df[base_numerical_features + base_categorical_features + ['Status']]

# Handle missing values
print("\nHandling missing values...")
num_imputer = SimpleImputer(strategy='median')
df[base_numerical_features] = num_imputer.fit_transform(df[base_numerical_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[base_categorical_features] = cat_imputer.fit_transform(df[base_categorical_features])

# Create derived features
print("\nCreating derived features...")
df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
df['monthly_payment'] = (df['loan_amount'] * (0.06/12) * (1 + (0.06/12))**(df['term'])) / ((1 + (0.06/12))**(df['term']) - 1)
df['payment_to_income'] = (df['monthly_payment'] * 12) / (df['income'] + 1)
df['credit_score_scaled'] = df['Credit_Score'] / 850
df['disposable_income'] = (df['income']/12) * (1 - df['dtir1']/100) - df['monthly_payment']

# Create risk indicators
print("\nCalculating risk indicators...")
df['high_dti'] = pd.cut(df['dtir1'], 
                        bins=[-float('inf'), 36, 43, 50, float('inf')],
                        labels=[0, 1, 2, 3]).fillna(3).astype(int)

df['high_loan_to_income'] = pd.cut(df['loan_to_income'],
                                  bins=[-float('inf'), 2.5, 3, 4, float('inf')],
                                  labels=[0, 1, 2, 3]).fillna(3).astype(int)

df['credit_risk'] = pd.cut(df['Credit_Score'],
                          bins=[-float('inf'), 580, 640, 700, 850],
                          labels=[3, 2, 1, 0]).fillna(3).astype(int)

# Define final features
numerical_features = [
    'loan_amount', 'income', 'Credit_Score', 'dtir1',
    'loan_to_income', 'payment_to_income', 'monthly_payment',
    'disposable_income', 'credit_score_scaled',
    'high_dti', 'high_loan_to_income', 'credit_risk'
]

# Prepare data for modeling
X = df[numerical_features + base_categorical_features]
y = df['Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numerical_features]),
    columns=numerical_features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test[numerical_features]),
    columns=numerical_features,
    index=X_test.index
)

# Process categorical features
le_dict = {}
for feature in base_categorical_features:
    le_dict[feature] = preprocessing.LabelEncoder()
    X_train_scaled[feature] = le_dict[feature].fit_transform(X_train[feature])
    X_test_scaled[feature] = le_dict[feature].transform(X_test[feature])

# Find best k using cross-validation
print("\nFinding optimal k...")
k_values = [3, 5, 7, 9, 11]
best_k = None
best_score = 0

for k in k_values:
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='manhattan'
    )
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    mean_score = scores.mean()
    print(f"k={k}, F1 Score={mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nBest k: {best_k} (F1 Score: {best_score:.4f})")

# Train final model
print("\nTraining final model...")
final_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    metric='manhattan'
)
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)

print("\nTest Set Evaluation:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_loan_default(input_data):
    """
    Make a prediction using the trained KNN model.
    """
    # Create a DataFrame with the same structure as training data
    input_df = pd.DataFrame([input_data])
    
    print("\nProcessing input data...")
    print("Original input:", input_data)
    
    # Calculate derived features
    input_df['loan_to_income'] = input_df['loan_amount'] / (input_df['income'] + 1)
    input_df['monthly_payment'] = (input_df['loan_amount'] * (0.06/12) * (1 + (0.06/12))**(input_df['term'])) / ((1 + (0.06/12))**(input_df['term']) - 1)
    input_df['payment_to_income'] = (input_df['monthly_payment'] * 12) / (input_df['income'] + 1)
    input_df['credit_score_scaled'] = input_df['Credit_Score'] / 850
    input_df['loan_term_years'] = input_df['term'] / 12
    input_df['income_per_month'] = input_df['income'] / 12
    input_df['disposable_income'] = input_df['income_per_month'] * (1 - input_df['dtir1']/100) - input_df['monthly_payment']
    
    # Create risk indicators
    input_df['high_dti'] = pd.cut(input_df['dtir1'],
                                 bins=[-float('inf'), 36, 43, 50, float('inf')],
                                 labels=[0, 1, 2, 3]).fillna(3).astype(int)
    input_df['high_loan_to_income'] = pd.cut(input_df['loan_to_income'],
                                            bins=[-float('inf'), 2.5, 3, 4, float('inf')],
                                            labels=[0, 1, 2, 3]).fillna(3).astype(int)
    input_df['credit_risk'] = pd.cut(input_df['Credit_Score'],
                                    bins=[-float('inf'), 580, 640, 700, 850],
                                    labels=[3, 2, 1, 0]).fillna(3).astype(int)
    
    # Process features
    print("\nProcessing features...")
    numerical_input = input_df[numerical_features].copy()
    print("Numerical features before scaling:", numerical_input.iloc[0].to_dict())
    
    numerical_scaled = pd.DataFrame(
        scaler.transform(numerical_input),
        columns=numerical_features
    )
    print("Numerical features after scaling:", numerical_scaled.iloc[0].to_dict())
    
    # Process categorical features
    for feature in base_categorical_features:
        try:
            numerical_scaled[feature] = le_dict[feature].transform([input_df[feature].iloc[0]])
        except ValueError as e:
            print(f"Error processing {feature}: {e}")
            raise
    
    print("\nMaking prediction using model...")
    print(f"Model parameters: {final_model.get_params()}")
    
    # Make prediction
    prediction = final_model.predict(numerical_scaled)
    probability = final_model.predict_proba(numerical_scaled)
    
    # Fix interpretation: probability[0][1] is probability of default
    # If probability of default > 0.5, prediction should be "Default"
    predicted_default = probability[0][1] > 0.5
    
    print(f"\nRaw prediction: {'Default' if predicted_default else 'No Default'}")
    print(f"Probability distribution: No Default: {probability[0][0]:.4f}, Default: {probability[0][1]:.4f}")
    
    # Get nearest neighbors for explanation
    distances, indices = final_model.kneighbors(numerical_scaled)
    print(f"\nNearest neighbor distances: {distances[0]}")
    
    print("\nFeature Analysis:")
    print("\nKey Financial Metrics:")
    print(f"Loan Amount: ${input_df['loan_amount'].iloc[0]:,.2f}")
    print(f"Annual Income: ${input_df['income'].iloc[0]:,.2f}")
    print(f"Monthly Payment: ${input_df['monthly_payment'].iloc[0]:,.2f}")
    print(f"Disposable Income: ${input_df['disposable_income'].iloc[0]:,.2f}")
    print(f"Credit Score: {input_df['Credit_Score'].iloc[0]}")
    
    print("\nRisk Ratios:")
    print(f"Loan-to-Income: {input_df['loan_to_income'].iloc[0]:.2f}x")
    print(f"Payment-to-Income: {input_df['payment_to_income'].iloc[0]*100:.1f}%")
    print(f"DTI Ratio: {input_df['dtir1'].iloc[0]:.1f}%")
    
    print("\nRisk Assessment:")
    dti_level = input_df['high_dti'].iloc[0]
    lti_level = input_df['high_loan_to_income'].iloc[0]
    credit_level = input_df['credit_risk'].iloc[0]
    
    risk_levels = ['Low', 'Moderate', 'High', 'Very High']
    print(f"DTI Risk: {risk_levels[dti_level]}")
    print(f"Loan-to-Income Risk: {risk_levels[lti_level]}")
    print(f"Credit Risk: {risk_levels[credit_level]}")
    
    risk_score = ((dti_level + lti_level + credit_level) / 9 * 100)
    print(f"\nOverall Risk Score: {risk_score:.1f}/100")
    
    return predicted_default, probability[0]

# Example usage of prediction function
print("\nTesting model with example prediction...")
sample_input = {
    'loan_amount': 350000,  # More realistic loan amount
    'term': 360,           # 30-year term
    'income': 42000,       # Annual income
    'Credit_Score': 652,
    'dtir1': 50,
    'Gender': 'Male',
    'loan_type': 'type1',
    'loan_purpose': 'p1',
    'Credit_Worthiness': 'l1',
    'occupancy_type': 'pr'
}

try:
    print("\nMaking example prediction...")
    pred, prob = predict_loan_default(sample_input)
    print("\nExample Prediction Results:")
    print("===================")
    print(f"Prediction: {'Default' if pred else 'No Default'}")
    print(f"Probability of Default: {prob[1]:.4f}")
    print(f"Probability of No Default: {prob[0]:.4f}")
except Exception as e:
    print(f"Error in example prediction: {e}")
    raise

def get_user_input():
    print("\n\nInteractive Loan Default Prediction")
    print("===================================")
    
    user_input = {}
    
    # Get numerical features
    print("\nEnter numerical values:")
    print("\nNote: These values significantly impact your loan application.")
    for feature in ['loan_amount', 'term', 'income', 'Credit_Score']:
        while True:
            try:
                if feature == 'loan_amount':
                    value = float(input("Enter loan amount ($): "))
                elif feature == 'term':
                    value = float(input("Enter loan term (in months, e.g., 360 for 30 years): "))
                elif feature == 'income':
                    value = float(input("Enter annual income ($): "))
                    monthly_income = value / 12
                    print(f"Monthly income: ${monthly_income:,.2f}")
                elif feature == 'Credit_Score':
                    value = float(input("Enter credit score (300-850): "))
                
                if feature == 'Credit_Score' and (value < 300 or value > 850):
                    print("Credit Score should be between 300 and 850")
                    continue
                if value < 0:
                    print("Value cannot be negative")
                    continue
                user_input[feature] = value
                break
            except ValueError:
                print("Please enter a valid number")
    
    # Get monthly debt obligations
    print("\nMonthly Debt Obligations:")
    print("Please enter your current monthly debt payments (excluding the new loan):")
    while True:
        try:
            monthly_debt = float(input("Enter total monthly debt payments ($): "))
            if monthly_debt < 0:
                print("Monthly debt cannot be negative")
                continue
                
            # Calculate monthly income and DTI
            monthly_income = user_input['income'] / 12
            
            # Calculate estimated new monthly payment for the loan
            r = 0.06 / 12  # Monthly interest rate (6% annual)
            n = user_input['term']  # Number of months
            new_monthly_payment = (user_input['loan_amount'] * r * (1 + r)**n) / ((1 + r)**n - 1)
            
            # Calculate total monthly debt including new loan payment
            total_monthly_debt = monthly_debt + new_monthly_payment
            
            # Calculate DTI ratio (as a percentage)
            dti = (total_monthly_debt / monthly_income) * 100
            
            print(f"\nDTI Calculation Summary:")
            print(f"Current monthly debt: ${monthly_debt:,.2f}")
            print(f"Estimated new loan payment: ${new_monthly_payment:,.2f}")
            print(f"Total monthly debt: ${total_monthly_debt:,.2f}")
            print(f"Monthly income: ${monthly_income:,.2f}")
            print(f"Calculated DTI ratio: {dti:.1f}%")
            
            if dti > 100:
                print("\nWarning: DTI ratio exceeds 100%. This means monthly debts exceed monthly income.")
                proceed = input("Do you want to proceed with this DTI? (yes/no): ").lower()
                if proceed != 'yes':
                    continue
            
            user_input['dtir1'] = dti
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get categorical features
    print("\nEnter categorical values:")
    
    # Gender
    while True:
        gender = input("\nEnter Gender (Male/Female/Other): ").capitalize()
        if gender in ['Male', 'Female', 'Other']:
            user_input['Gender'] = gender
            break
        print("Invalid gender. Please enter Male, Female, or Other")
    
    # Loan Type
    print("\nLoan Types:")
    print("type1: Conventional Loan (traditional mortgage)")
    print("type2: FHA/VA/Special Program Loan")
    while True:
        loan_type = input("Enter loan type (type1/type2): ").lower()
        if loan_type in ['type1', 'type2']:
            user_input['loan_type'] = loan_type
            break
        print("Invalid loan type. Please enter type1 or type2")
    
    # Loan Purpose
    print("\nLoan Purpose:")
    print("p1: Home Purchase")
    print("p2: Refinancing")
    print("p3: Home Improvement")
    print("p4: Other")
    while True:
        loan_purpose = input("Enter loan purpose (p1/p2/p3/p4): ").lower()
        if loan_purpose in ['p1', 'p2', 'p3', 'p4']:
            user_input['loan_purpose'] = loan_purpose
            break
        print("Invalid loan purpose. Please enter p1, p2, p3, or p4")
    
    # Credit Worthiness
    print("\nCredit Worthiness Level:")
    print("l1: Prime (Good credit history)")
    print("l2: Subprime (Challenged credit history)")
    while True:
        credit_worthiness = input("Enter Credit Worthiness (l1/l2): ").lower()
        if credit_worthiness in ['l1', 'l2']:
            user_input['Credit_Worthiness'] = credit_worthiness
            break
        print("Invalid credit worthiness. Please enter l1 or l2")
    
    # Occupancy Type
    print("\nOccupancy Type:")
    print("pr: Primary Residence (You'll live there)")
    print("sr: Secondary Residence (Vacation/Second home)")
    print("ir: Investment Property (Rental/Investment)")
    while True:
        occupancy = input("Enter occupancy type (pr/sr/ir): ").lower()
        if occupancy in ['pr', 'sr', 'ir']:
            user_input['occupancy_type'] = occupancy
            break
        print("Invalid occupancy type. Please enter pr, sr, or ir")
    
    return user_input

def interactive_prediction():
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            # Make prediction
            pred, prob = predict_loan_default(user_input)
            
            # Print results with corrected interpretation
            print("\nPrediction Results:")
            print("===================")
            print(f"Prediction: {'Default' if pred else 'No Default'}")
            print(f"Probability of Default: {prob[1]:.4f}")
            print(f"Probability of No Default: {prob[0]:.4f}")
            
            # Add clear risk interpretation
            if prob[1] > 0.8:
                print("\nRISK ASSESSMENT: Very High Risk of Default")
            elif prob[1] > 0.6:
                print("\nRISK ASSESSMENT: High Risk of Default")
            elif prob[1] > 0.4:
                print("\nRISK ASSESSMENT: Moderate Risk of Default")
            elif prob[1] > 0.2:
                print("\nRISK ASSESSMENT: Low Risk of Default")
            else:
                print("\nRISK ASSESSMENT: Very Low Risk of Default")
            
            # Ask if user wants to make another prediction
            again = input("\nWould you like to make another prediction? (yes/no): ").lower()
            if again != 'yes':
                break
                
        except ValueError as e:
            print(f"\nError: {e}")
            print("Please try again with valid inputs")
            continue
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again")
            continue

# Run the interactive prediction system
if __name__ == "__main__":
    print("\nWelcome to the Loan Default Prediction System")
    interactive_prediction()
