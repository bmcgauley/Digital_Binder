import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("Raw_Full_Titanic_Data.csv")

# Feature Engineering - Keep important features and create new ones
# Instead of dropping PassengerId, Name, Ticket, and Cabin, extract information

# Extract title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Group rare titles
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Extract cabin letter (deck information)
df['Deck'] = df['Cabin'].str.slice(0, 1)
df['Deck'] = df['Deck'].fillna('U')  # U for unknown

# Extract family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create isAlone feature
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], 
                       labels=['Child', 'Teenager', 'Adult', 'Elderly'])

# Create fare category
df['FareCategory'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 4, 
                            labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Feature for embarked * pclass interaction
df['Embarked_Pclass'] = df['Embarked'].astype(str) + df['Pclass'].astype(str)

# Print missing values
print("Missing values before imputation:")
print(df.isnull().sum())

# Impute missing values instead of dropping rows
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df['Age'].values.reshape(-1, 1))

embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

fare_imputer = SimpleImputer(strategy='median')
df['Fare'] = fare_imputer.fit_transform(df['Fare'].values.reshape(-1, 1))

# Verify missing values after imputation
print("\nMissing values after imputation:")
print(df.isnull().sum())

plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], kde=True)
plt.title('Distribution of Fare')
plt.show()
# Now encode categorical features
label_encoders = {}
categorical_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup', 'FareCategory', 'Embarked_Pclass']

# Check for outliers
plt.figure(figsize=(8, 6))
sns.boxplot(df['Fare'])
plt.title('Boxplot of Fare')
plt.show()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Drop columns we don't need anymore
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Handle outliers in 'Fare' using capping instead of removal
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers instead of removing rows
df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])
df['Fare'] = np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])

# Store Survived column before normalization
survived_column = df['Survived']

# Remove the 'Survived' feature for normalization
df_for_norm = df.drop(['Survived'], axis=1)

# Normalize numerical features using Min-Max scaling
scaler = MinMaxScaler()
cols_to_scale = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
df_for_norm[cols_to_scale] = scaler.fit_transform(df_for_norm[cols_to_scale])

# Add the Survived feature back
df_final = pd.concat([survived_column, df_for_norm], axis=1)

# Print the final feature set
print("\nFinal feature set:")
print(df_final.columns.tolist())

# Split data for model training
X = df_final.drop('Survived', axis=1)
y = df_final['Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Create and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Feature importance analysis
# Calculate correlation with Survived for feature importance
corr_with_target = df_final.corr()['Survived'].sort_values(ascending=False)
print("\nFeature Correlation with Survival:")
print(corr_with_target)