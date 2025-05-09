# Feature Engineering Lab 1 - Python Outline
# Author: Brian McGauley, Mohannad Yasin
# Date: 02/01/2021

# ===============================
# 1. Importing Required Libraries
# ===============================
# Import pandas for data manipulation
# Import numpy for numerical operations
# Import LabelEncoder and preprocessing from sklearn for encoding and normalization
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing

# Function to print data retention stats
def print_retention_stats(initial_rows, remaining_rows):
    rows_removed = initial_rows - remaining_rows
    retention_pct = (remaining_rows / initial_rows) * 100
    
    print(f"\nData Retention Statistics:")
    print(f"Initial number of rows: {initial_rows:,}")
    print(f"Remaining rows: {remaining_rows:,}")
    print(f"Rows removed: {rows_removed:,}")
    print(f"Percentage of data retained: {retention_pct:.2f}%")

print("\nStep #1 - Completed")
# ===============================
# 2. Loading the Dataset
# ===============================
# Load the dataset 'raw_diabetes_string included.csv' from the Sample Data folder
# Optionally, load the dataset from Google Drive if working in Google Colab
df = pd.read_csv('diabetes.csv')
print("\nStep #2 - Completed")
# ===============================
# 3. Handling Missing or Null Values
# ===============================
# Check for missing or null values in the following columns
checkColumns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Print initial null value counts
print("\nStep #3")
print("Initial null values:")
print(df[checkColumns].isnull().sum())
print(f"\nInitial number of rows: {len(df)}")

# Drop rows with missing values since we have plenty of data
initial_rows = len(df)
df = df.dropna()
print_retention_stats(initial_rows, len(df))

print("\nStep #3.2")
# Preview of data after handling missing values
print("\nPreview of data after handling missing values:")
print(df.head())


# ===============================
# 4. String to Integer Conversion
# ===============================
# Convert the 'Outcome' column from categorical to numeric using LabelEncoder
# - Encode the 'Outcome' column
# - Drop the original 'Outcome' column if necessary
# - Add the encoded column back to the dataset

print("\nStep #4")
# Create a LabelEncoder instance
le = LabelEncoder()



# Create a copy of the Outcome column and encode it
df['Outcome_encoded'] = le.fit_transform(df['Outcome'])

# Drop the original Outcome column and rename the encoded column
df = df.drop('Outcome', axis=1)
df = df.rename(columns={'Outcome_encoded': 'Outcome'})

# Print preview of the data after encoding
print("\nPreview of data after string to integer conversion:")
print(df.head())

# ===============================
# 5. Data Normalization
# ===============================
# Normalize the feature columns (excluding the 'Outcome' column)
# - Use StandardScaler or MinMaxScaler for normalization
# - Create a new DataFrame with the normalized data
# - Add the 'Outcome' column back to the normalized DataFrame

print("\nStep #5")


# Separate features and target
features = df.drop('Outcome', axis=1)
outcome = df['Outcome']

# Create and fit the StandardScaler
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Convert normalized features back to DataFrame with column names
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)

# Add the Outcome column back
df_normalized['Outcome'] = outcome

# Update our main dataframe
df = df_normalized

# Print preview of normalized data
print("\nPreview of normalized data:")
print(df.head())
print("\nSummary statistics of normalized data:")
print(df.describe())

# ===============================
# 6. Column Manipulation
# ===============================
# Move two columns to different positions within the DataFrame
# - Use reindex or insert methods to rearrange columns
# Rename two column headers to new names
# - Use the rename() function for column renaming

print("\nStep #6")

# Print current column order
print("\nOriginal column order:")
print(df.columns.tolist())

# Rearrange columns: move 'Age' to the beginning and 'BMI' after 'Age'
cols = df.columns.tolist()
cols.remove('Age')
cols.remove('BMI')
new_cols = ['Age', 'BMI'] + cols

df = df.reindex(columns=new_cols)

# Print column order after rearrangement
print("\nColumn order after moving 'Age' and 'BMI':")
print(df.columns.tolist())

# Rename 'DiabetesPedigreeFunction' to 'Pedigree' and 'BloodPressure' to 'BP'
df = df.rename(columns={
    'DiabetesPedigreeFunction': 'Pedigree',
    'BloodPressure': 'BP'
})

# Print preview after renaming
print("\nPreview after column renaming:")
print(df.head())
print("\nFinal column names:")
print(df.columns.tolist())

# ===============================
# 7. Applying Similar Maneuvers to Additional Datasets
# ===============================

print("\nStep #7 - Gapminder Dataset")
# Step 1 - Loading the Dataset
print("\nStep #7.1 - Loading the Dataset")
# Load the first dataset (gapminder)
df_gap = pd.read_csv('gapminder.csv')

# Add preview of raw data
print("\nPreview of raw gapminder data:")
print(df_gap.head())
print("\nUnique continents:", df_gap['continent'].unique())
print("\nFirst 5 unique countries:", df_gap['country'].unique()[:5])

# Step 2 - Handling Missing Values
print("\nStep #7.2 - Handling Missing Values")
print("Initial null values in gapminder:")
print(df_gap.isnull().sum())
print(f"\nInitial number of rows: {len(df_gap)}")

# Drop any rows with missing values
initial_rows = len(df_gap)
df_gap = df_gap.dropna()
print_retention_stats(initial_rows, len(df_gap))

# Step 3 - String to Integer Conversion
print("\nStep #7.3 - String to Integer Conversion")
# Create LabelEncoder instance
le = LabelEncoder()

# Encode 'continent' and 'country' columns (replace original columns)
df_gap['continent'] = le.fit_transform(df_gap['continent'])
df_gap['country'] = le.fit_transform(df_gap['country'])

# Print statistics about the encoded data
print("\nEncoding Statistics:")
print(f"Number of unique countries:\n {df_gap['country'].nunique()}")
print(f"Number of unique continents:\n {df_gap['continent'].nunique()}")
print("\nSample encoding mapping:")
print("\nFirst 5 Continents:\n")
for i, cont in enumerate(sorted(df_gap['continent'].unique())):
    
    print(f"Code {cont}: {le.inverse_transform([cont])[0]}")
print("\nFirst 5 Countries:\n")
for i, country in enumerate(sorted(df_gap['country'].unique())[:5]):
    print(f"Code {country}: {le.inverse_transform([country])[0]}")


print("\nPreview after encoding:")
print(df_gap.head())

# Step 4 - Data Normalization
print("\nStep #7.4 - Data Normalization")
# Separate numeric columns for normalization (excluding year which is categorical)
numeric_columns = ['lifeExp', 'pop', 'gdpPercap']




# Create and fit the StandardScaler
scaler = StandardScaler()
df_gap[numeric_columns] = scaler.fit_transform(df_gap[numeric_columns])

print("\nPreview of normalized data:")
print(df_gap.head())

# Step 5 - Column Manipulation
print("\nStep #7.5 - Column Manipulation")
# Print current column order
print("\nOriginal column order:")
print(df_gap.columns.tolist())



# Rearrange columns: move 'year' to beginning and 'gdpPercap' after it
cols = df_gap.columns.tolist()
cols.remove('year')
cols.remove('gdpPercap')
new_cols = ['year', 'gdpPercap'] + cols

df_gap = df_gap.reindex(columns=new_cols)

# Rename columns
df_gap = df_gap.rename(columns={
    'lifeExp': 'LifeExpectancy',
    'gdpPercap': 'GDPperCapita'
})

print("\nFinal preview of gapminder data:")
print(df_gap.head())
print("\nFinal column names:")
print(df_gap.columns.tolist())

# ===============================
#8. Third Dataset - Random Survey
# ===============================
print("\nStep #8 - Random Survey Dataset")

# Step 1 - Loading the Dataset
print("\nStep #8.1 - Loading the Dataset")
# Load the third dataset - using encoding that works with Excel files
df_survey = pd.read_csv('rnd_survey.csv', encoding='latin-1')

# Add preview of raw data
print("\nPreview of raw survey data:")
print(df_survey.head())
print("\nColumns in dataset:", df_survey.columns.tolist())

# For survey dataset, analyze missing values first
print("\nMissing value counts per column:")

# Preview missing values before handling
print("\nMissing null values before handling:")
print(df_survey.isnull().sum())

print("\nStep #8.2 - Handling Missing Values")
# Store initial row count
initial_rows = len(df_survey)

# explicitly list columns we know are numeric/categorical
numeric_columns = ['Year', 'RD_Value', 'Relative_Sampling_Error']
categorical_columns = ['Variable', 'Breakdown', 'Breakdown_category', 'Status', 'Unit', 'Footnotes']

# Handle missing values - simpler approach with type checking
for column in df_survey.columns:
    # Check if column has any missing values first
    if df_survey[column].isnull().sum() > 0:
        try:
            # Try to convert to numeric and use mean
            df_survey[column] = pd.to_numeric(df_survey[column])
            df_survey[column] = df_survey[column].fillna(df_survey[column].mean())
        except:
            # If conversion fails, treat as categorical and use mode
            df_survey[column] = df_survey[column].fillna(df_survey[column].mode()[0])

# Print retention statistics
print_retention_stats(initial_rows, len(df_survey))

# Verify no missing values remain
print("\nRemaining null values after handling:")
print(df_survey.isnull().sum())

# Step 3 - String to Integer Conversion
print("\nStep #8.3 - String to Integer Conversion")
# Create LabelEncoder instance
le = LabelEncoder()

# Find categorical columns (assuming string/object dtype columns are categorical)
categorical_columns = df_survey.select_dtypes(include=['object']).columns

# Encode all categorical columns
for col in categorical_columns:
    df_survey[col] = le.fit_transform(df_survey[col])

# Print statistics about the encoded data
print("\nEncoding Statistics:")
print("Number of categorical columns encoded:", len(categorical_columns))
print("\nEncoded columns:", categorical_columns.tolist())

print("\nPreview after encoding:")
print(df_survey.head())

# Step 4 - Data Normalization
print("\nStep #8.4 - Data Normalization")
# Separate numeric columns for normalization
numeric_columns = df_survey.select_dtypes(include=['float64', 'int64']).columns

# Create and fit the StandardScaler
scaler = StandardScaler()
df_survey[numeric_columns] = scaler.fit_transform(df_survey[numeric_columns])

print("\nPreview of normalized data:")
print(df_survey.head())

# Step 5 - Column Manipulation
print("\nStep #8.5 - Column Manipulation")
# Print current column order
print("\nOriginal column order:")
print(df_survey.columns.tolist())

# Rearrange columns: move Year, RD_Value to the beginning
cols = df_survey.columns.tolist()
cols.remove('Year')
cols.remove('RD_Value')
new_cols = ['Year', 'RD_Value'] + cols

df_survey = df_survey.reindex(columns=new_cols)

# Rename columns for clarity
df_survey = df_survey.rename(columns={
    'RD_Value': 'Research_Development_Value',
    'Relative_Sampling_Error': 'Sampling_Error',
    'Breakdown_category': 'Category'
})

print("\nFinal preview of survey data:")
print(df_survey.head())
print("\nFinal column names:")
print(df_survey.columns.tolist())

# ===============================
# 9. Saving and Submitting Work
# ===============================
# Save the final DataFrames to .csv files
# Ensure three completed .ipynb files are ready for submission
# Each team member must submit their own copy on Canvas before the due date
print("\nStep #9 - Saving Processed Data")

# Save the processed diabetes dataset
df.to_csv('processed_diabetes.csv', index=False)
print("\nProcessed diabetes dataset saved to: processed_diabetes.csv")

# Save the processed gapminder dataset
df_gap.to_csv('processed_gapminder.csv', index=False)
print("Processed gapminder dataset saved to: processed_gapminder.csv")

print("\nFeature Engineering Lab 1 - Complete!")

