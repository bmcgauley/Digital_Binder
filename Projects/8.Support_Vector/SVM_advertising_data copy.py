import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
# Load the data
df = pd.read_csv('advertising.csv')
# Remove 'Ad Topic Line' and 'Timestamp' features
df = df.drop(['Ad Topic Line', 'Timestamp'], axis=1)

le_country = LabelEncoder()
le_city = LabelEncoder()

df['Country'] = le_country.fit_transform(df['Country'])
df['City'] = le_city.fit_transform(df['City'])
df.head()
# Split between featues as X and target as y

X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']
# Apply StandardScaler to the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Set the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Set the model
model = SVC()

#Use One-vs-Rest (OvA) strategy
#model = OneVsRestClassifier(SVC())

#Use One-vs-One (OvO) strategy
#model = OneVsOneClassifier(SVC())
# model fit for the data train
model.fit(X_train, y_train)
# Make predictions on the test data
model_predict = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, model_predict)
print("\nAccuracy:", accuracy)
# Run precision, recall, and f1 score

# Generate the classification report
report = classification_report(y_test, model_predict)
print("\nClassification Report:\n", report)

# Extract precision, recall, and F1-score from the report
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, model_predict)
recall = recall_score(y_test, model_predict)
f1 = f1_score(y_test, model_predict)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)




# Function to get new data point with proper feature handling
def get_new_data_point():
    """Get new data point from user with proper feature handling"""
    print("\nEnter new data for prediction:")
    
    # Numerical features
    daily_time = float(input("Enter Daily Time Spent on Site: "))
    age = int(input("Enter Age: "))
    area_income = float(input("Enter Area Income: "))
    daily_internet = float(input("Enter Daily Internet Usage: "))
    male = int(input("Enter Male (1 for Male, 0 for Female): "))
    
    # Categorical features
    country = input("Enter Country: ")
    city = input("Enter City: ")
    
    # Create DataFrame with proper column names matching the original data
    new_data = pd.DataFrame({
        'Daily Time Spent on Site': [daily_time],
        'Age': [age],
        'Area Income': [area_income],
        'Daily Internet Usage': [daily_internet],
        'Male': [male],
        'Country': [country],
        'City': [city]
    })
    
    # Encode categorical variables with error handling
    try:
        new_data['Country'] = le_country.transform(new_data['Country'])
    except ValueError:
        print(f"\nError: Country '{country}' not found in training data.")
        print("Available countries:", sorted(le_country.classes_))
        raise ValueError("Invalid country")
        
    try:
        new_data['City'] = le_city.transform(new_data['City'])
    except ValueError:
        print(f"\nError: City '{city}' not found in training data.")
        print("Available cities:", sorted(le_city.classes_))
        raise ValueError("Invalid city")
    
    return new_data

# Example usage
while True:
    try:
        new_data = get_new_data_point()
        
        # Scale the new data and preserve feature names
        new_data_scaled = pd.DataFrame(
            scaler.transform(new_data),
            columns=new_data.columns
        )
        
        # Make prediction
        prediction = model.predict(new_data_scaled)[0]
        probabilities = model.predict_proba(new_data_scaled)[0]
        
        print("\nPrediction:", "Clicked on Ad" if prediction == 1 else "Did not click on Ad")
        print("Prediction probabilities:", probabilities)
        
        retry = input("\nDo you want to make another prediction? (y/n): ")
        if retry.lower() != 'y':
            break
            
    except ValueError as e:
        print("\nPlease try again with valid values.")
        retry = input("Do you want to start over? (y/n): ")
        if retry.lower() != 'y':
            break
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        retry = input("Do you want to start over? (y/n): ")
        if retry.lower() != 'y':
            break
