import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Raw_Full_Titanic_Data.csv")
df.head()

#drop PassengerId, Name, Ticket, and Cabin

df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df.head()

# drop missing and null values

# Check for missing values
print(df.isnull().sum())

# Drop rows with any missing values
df.dropna(inplace=True)

# Verify that there are no more missing values
print(df.isnull().sum())

# Convert the following categorical features to numeric using LabelEncoder
le = LabelEncoder()

# Convert 'Sex' column to numeric
df['Sex'] = le.fit_transform(df['Sex'])

# Convert 'Embarked' column to numeric
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str)) #astype(str) handles potential NaN values

df.head()

# Check the distribution of the 'Fare' feature
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], kde=True)
plt.title('Distribution of Fare')
plt.show()

# Check for outliers
plt.figure(figsize=(8, 6))
sns.boxplot(df['Fare'])
plt.title('Boxplot of Fare')
plt.show()


# the boxplot is for the 'Fare' column
# Get the quartiles
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find the outliers
outliers = df[(df['Fare'] < lower_bound) | (df['Fare'] > upper_bound)]['Fare']

# Print the outliers
outliers

# prompt: Let's remove the entire row where its Fare value is an outlier

# Remove rows where 'Fare' is an outlier
df_no_outliers = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# Print some info
print("Original DataFrame shape:", df.shape)
print("DataFrame shape after removing outliers:", df_no_outliers.shape)

df_no_outliers.head()

# remove the Survived feature and run data normalization with the rest of features

import pandas as pd
# Remove the 'Survived' feature
df_no_outliers = df_no_outliers.drop(['Survived'], axis=1)

# Normalize the remaining features using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df_no_outliers.columns)

df_normalized.head()

# Add the Survived feature back with the the rest of features

import pandas as pd
# Concatenate 'Survived' back to the DataFrame
df_final = pd.concat([df['Survived'], df_normalized], axis=1)
df_final.head()

# split the data_final setting up for naive bayes algorithm model

from sklearn.model_selection import train_test_split

# Assuming df_final is your DataFrame
X = df_final.drop('Survived', axis=1)  # Features
y = df_final['Survived']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% test

# Now you have X_train, X_test, y_train, and y_test for your Naive Bayes model
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# check for missing or null values, delete them all.

# Check for missing values again after all the operations
print(df_final.isnull().sum())

# Drop rows with any missing values in the final dataframe
df_final.dropna(inplace=True)

# Verify that there are no more missing values
print(df_final.isnull().sum())

# Split the data_final setting up for naive bayes algorithm model
X = df_final.drop('Survived', axis=1)  # Features
y = df_final['Survived']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% test

# Now you have X_train, X_test, y_train, and y_test for your Naive Bayes model
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#This is model generation

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

#This is to evaluate the model
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Evaluation metrics with precision, recall, and f1 score

from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming you have y_test and y_pred from the previous code

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1}")