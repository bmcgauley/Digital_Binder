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

# Store the original column names before encoding
original_columns = df.drop('Clicked on Ad', axis=1).columns.tolist()

le_country = LabelEncoder()
le_city = LabelEncoder()

df['Country'] = le_country.fit_transform(df['Country'])
df['City'] = le_city.fit_transform(df['City'])

# Split between features as X and target as y
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Apply StandardScaler to the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Set the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the model
model = SVC(probability=True)  # Added probability=True to enable predict_proba
model.fit(X_train, y_train)

# Make predictions on the test data
model_predict = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, model_predict)
print("\nAccuracy:", accuracy)

# Generate the classification report
report = classification_report(y_test, model_predict)
print("\nClassification Report:\n", report)

# Extract precision, recall, and F1-score from the report
precision = precision_score(y_test, model_predict)
recall = recall_score(y_test, model_predict)
f1 = f1_score(y_test, model_predict)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

def get_new_data_point():
    """Prompts the user to input a new data point and returns it as a DataFrame."""
    print("\nEnter new data for prediction:")
    
    try:
        daily_time_spent_on_site = float(input("Enter Daily Time Spent on Site: "))
        age = float(input("Enter Age: "))
        area_income = float(input("Enter Area Income: "))
        daily_internet_usage = float(input("Enter Daily Internet Usage: "))
        male = int(input("Enter Male (1 for Male, 0 for Female): "))
        country = input("Enter Country: ")
        city = input("Enter City: ")

        # Create DataFrame with exact same column names as original data
        new_data = pd.DataFrame({
            'Daily Time Spent on Site': [daily_time_spent_on_site],
            'Age': [age],
            'Area Income': [area_income],
            'Daily Internet Usage': [daily_internet_usage],
            'Male': [male],
            'Country': [country],
            'City': [city]
        })
        
        # Encode categorical variables
        new_data['Country'] = le_country.transform(new_data['Country'])
        new_data['City'] = le_city.transform(new_data['City'])
        
        # Ensure columns are in the same order as original data
        new_data = new_data[original_columns]
        
        return new_data

    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return None
    except KeyError:
        print("Invalid input. Country or City not in training data.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
while True:
    new_data = get_new_data_point()
    
    if new_data is not None:
        # Scale the new data point
        new_data_scaled = scaler.transform(new_data)
        
        # Predict the output
        prediction = model.predict(new_data_scaled)[0]
        probabilities = model.predict_proba(new_data_scaled)[0]
        
        print("\nPrediction:", "Clicked on Ad" if prediction == 1 else "Did not click on Ad")
        print("Prediction probabilities:", probabilities)
    
    retry = input("\nDo you want to make another prediction? (y/n): ")
    if retry.lower() != 'y':
        break
