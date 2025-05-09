#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apartment Rental Price Prediction using Random Forest
This script analyzes what factors we can control to optimize rental listings.
Focus is on actionable insights for property managers and landlords.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz, plot_tree
import graphviz

# Set style for better visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load the dataset
print("\nStep 1: Loading dataset...")
from ucimlrepo import fetch_ucirepo
apartment_data = fetch_ucirepo(id=555)
df = pd.concat([apartment_data.data.features, apartment_data.data.targets], axis=1)
print("\nInitial dataset shape:", df.shape)

# Step 2: Data Cleaning and Preprocessing
print("\nStep 2: Data Cleaning and Preprocessing...")

# Convert numeric columns from object to float
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')

# Fill missing values first
df['square_feet'] = df['square_feet'].fillna(df['square_feet'].median())
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())

# Remove outliers (prices that are too high or too low)
q1 = df['price'].quantile(0.01)
q3 = df['price'].quantile(0.99)
df = df[(df['price'] >= q1) & (df['price'] <= q3)]

# Step 3: Feature Engineering - Separated by Controllable vs Uncontrollable Factors
print("\nStep 3: Feature Engineering - Focusing on Controllable Factors...")

# Controllable Features (things we can influence):
print("\nAnalyzing Controllable Features:")

print("1. Listing Presentation:")
# Better handling of text features
df['has_description'] = df['body'].notna().astype(int)
df['description_length'] = df['body'].str.len().fillna(0)
df['title_length'] = df['title'].str.len().fillna(0)
df['has_photo'] = df['has_photo'].fillna('false').astype(str).str.lower()
df['has_photo'] = (df['has_photo'] == 'true').astype(int)

print("2. Amenities and Policies:")
df['amenities_count'] = df['amenities'].str.count(',').fillna(0) + 1
df['pets_allowed'] = df['pets_allowed'].fillna('false').astype(str).str.lower()
df['pets_allowed'] = (df['pets_allowed'] == 'true').astype(int)

print("3. Timing Strategy:")
df['posting_date'] = pd.to_datetime(df['time'], unit='s')
df['posting_month'] = df['posting_date'].dt.month
df['posting_day_of_week'] = df['posting_date'].dt.dayofweek
df['is_weekend'] = df['posting_day_of_week'].isin([5, 6]).astype(int)

# Uncontrollable Features (for context):
print("\nTracking Uncontrollable Features:")
print("1. Location Factors:")
df['state_mean_price'] = df.groupby('state')['price'].transform('mean')
df['city_mean_price'] = df.groupby('cityname')['price'].transform('mean')

# Analyze impact of controllable features
print("\nAnalyzing Impact of Controllable Features:")

# 1. Impact of Listing Quality
print("\nListing Quality Analysis:")

# Print unique values for debugging
print("\nUnique values in has_photo:", df['has_photo'].unique())
print("Unique values in has_description:", df['has_description'].unique())

# Photos impact
print("\nPhoto Statistics:")
photo_counts = df['has_photo'].value_counts()
photo_means = df.groupby('has_photo')['price'].mean()

for val in sorted(df['has_photo'].unique()):
    count = photo_counts[val]
    mean_price = photo_means[val]
    status = "with" if val == 1 else "without"
    print(f"Listings {status} photos (count={count}):")
    print(f"Average price: ${mean_price:,.2f}")

if len(photo_means) == 2:
    photo_impact = ((photo_means[1] - photo_means[0]) / photo_means[0] * 100)
    print(f"Price difference: {photo_impact:.1f}%")

# Description impact
print("\nDescription Statistics:")
desc_counts = df['has_description'].value_counts()
desc_means = df.groupby('has_description')['price'].mean()

for val in sorted(df['has_description'].unique()):
    count = desc_counts[val]
    mean_price = desc_means[val]
    status = "with" if val == 1 else "without"
    print(f"Listings {status} descriptions (count={count}):")
    print(f"Average price: ${mean_price:,.2f}")

if len(desc_means) == 2:
    desc_impact = ((desc_means[1] - desc_means[0]) / desc_means[0] * 100)
    print(f"Price difference: {desc_impact:.1f}%")

# 2. Timing Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='posting_day_of_week', y='price', data=df)
plt.title('Price Distribution by Day of Week')
plt.xlabel('Day (0=Monday, 6=Sunday)')

plt.subplot(1, 2, 2)
sns.boxplot(x='posting_month', y='price', data=df)
plt.title('Price Distribution by Month')
plt.xlabel('Month')
plt.tight_layout()
plt.show()

# 3. Amenities Impact
plt.figure(figsize=(10, 6))
sns.regplot(x='amenities_count', y='price', data=df, scatter_kws={'alpha':0.5})
plt.title('Impact of Number of Amenities on Price')
plt.show()

# Select features for modeling, prioritizing controllable factors
controllable_features = [
    # Listing Presentation
    'has_photo', 'has_description', 'description_length', 'title_length',
    
    # Property Features
    'square_feet', 'bedrooms', 'bathrooms', 'amenities_count',
    
    # Policies
    'pets_allowed',
    
    # Timing
    'posting_month', 'posting_day_of_week', 'is_weekend'
]

context_features = [
    'state_mean_price', 'city_mean_price', 'latitude', 'longitude'
]

X = df[controllable_features + context_features]
y = df['price']

# Split and train as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Analyze feature importance specifically for controllable features
controllable_importance = pd.DataFrame({
    'feature': controllable_features,
    'importance': rf_model.feature_importances_[:len(controllable_features)]
})
controllable_importance = controllable_importance.sort_values('importance', ascending=False)

print("\nImportance of Controllable Features:")
print(controllable_importance)

# Visualize importance of controllable features
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=controllable_importance)
plt.title('Importance of Controllable Features for Price')
plt.tight_layout()
plt.show()

# Generate actionable recommendations
print("\nActionable Recommendations for Listing Optimization:")
print("1. Listing Presentation:")
top_presentation_features = controllable_importance[
    controllable_importance['feature'].isin(['has_photo', 'description_length', 'title_length'])
]
for _, row in top_presentation_features.iterrows():
    print(f"- {row['feature']}: Impact Score = {row['importance']:.4f}")

print("\n2. Property Features:")
top_property_features = controllable_importance[
    controllable_importance['feature'].isin(['square_feet', 'bedrooms', 'bathrooms', 'amenities_count'])
]
for _, row in top_property_features.iterrows():
    print(f"- {row['feature']}: Impact Score = {row['importance']:.4f}")

print("\n3. Timing Strategy:")
timing_features = controllable_importance[
    controllable_importance['feature'].isin(['posting_month', 'posting_day_of_week', 'is_weekend'])
]
for _, row in timing_features.iterrows():
    print(f"- {row['feature']}: Impact Score = {row['importance']:.4f}")

# Calculate optimal posting time
best_day = int(df.groupby('posting_day_of_week')['price'].mean().idxmax())
best_month = int(df.groupby('posting_month')['price'].mean().idxmax())

print(f"\nOptimal Posting Time:")
print(f"Best day to post: Day {best_day} (0=Monday, 6=Sunday)")
print(f"Best month to post: Month {best_month}")

# After training the model and before the interactive section, add:

print("\nVisualizing Random Forest Structure...")

# Visualize a single tree from the forest
plt.figure(figsize=(20,10))
plot_tree(rf_model.estimators_[0], 
          feature_names=X.columns,
          filled=True,
          max_depth=3,  # Limit depth for visibility
          fontsize=10)
plt.title("Sample Decision Tree from Random Forest")
plt.show()

# Create feature importance plot with error bars
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_,
    'std': np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
})
importances = importances.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.errorbar(x=range(len(importances)), 
            y=importances['importance'], 
            yerr=importances['std'], 
            fmt='o',
            capsize=5)
plt.xticks(range(len(importances)), importances['feature'], rotation=45, ha='right')
plt.title('Feature Importance with Standard Deviation across Trees')
plt.tight_layout()
plt.show()

# Visualize tree paths for different price ranges
print("\nAnalyzing Decision Paths for Different Price Ranges...")
sample_indices = [
    df[df['price'] < df['price'].quantile(0.25)].index[0],  # Low price
    df[df['price'].between(df['price'].quantile(0.45), df['price'].quantile(0.55))].index[0],  # Medium price
    df[df['price'] > df['price'].quantile(0.75)].index[0]  # High price
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
    price_category = ['Low', 'Medium', 'High'][idx]
    sample_price = df.loc[sample_idx, 'price']
    
    plot_tree(rf_model.estimators_[0],
              feature_names=X.columns,
              filled=True,
              max_depth=3,
              ax=ax)
    ax.set_title(f'{price_category} Price Example\n(${sample_price:,.2f})')

plt.tight_layout()
plt.show()

# Create a correlation matrix heatmap for feature relationships
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Interactive Section for User Input and Recommendations
print("\n" + "="*50)
print("Interactive Listing Analyzer")
print("="*50)
print("\nEnter your listing details to get predictions and recommendations:")

def get_numeric_input(prompt, min_val=0, max_val=float('inf'), allow_float=True):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value if allow_float else int(value)
            print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def get_yes_no_input(prompt):
    while True:
        response = input(prompt + " (yes/no): ").lower()
        if response in ['yes', 'y']:
            return 1
        elif response in ['no', 'n']:
            return 0
        print("Please enter 'yes' or 'no'")

try:
    # Get user input for listing details
    square_feet = get_numeric_input("Square feet: ", min_val=100)
    bedrooms = get_numeric_input("Number of bedrooms: ", min_val=0, max_val=10, allow_float=False)
    bathrooms = get_numeric_input("Number of bathrooms: ", min_val=0, max_val=10)
    
    has_photo = get_yes_no_input("Do you have photos for the listing?")
    has_description = get_yes_no_input("Do you have a description for the listing?")
    
    description_length = 0
    if has_description:
        description = input("Enter your listing description (or press Enter to skip): ")
        description_length = len(description)
    
    title = input("Enter your listing title: ")
    title_length = len(title)
    
    amenities = input("Enter amenities (comma-separated, or press Enter if none): ")
    amenities_count = len(amenities.split(',')) if amenities.strip() else 0
    
    pets_allowed = get_yes_no_input("Are pets allowed?")
    
    # Get location details
    latitude = get_numeric_input("Latitude (e.g., 37.7749): ", min_val=-90, max_val=90)
    longitude = get_numeric_input("Longitude (e.g., -122.4194): ", min_val=-180, max_val=180)
    
    # Use median values from the dataset for context features if not available
    state_mean_price = df['state_mean_price'].median()
    city_mean_price = df['city_mean_price'].median()
    
    # Current month and day
    from datetime import datetime
    current_date = datetime.now()
    posting_month = current_date.month
    posting_day_of_week = current_date.weekday()
    is_weekend = 1 if posting_day_of_week >= 5 else 0
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'has_photo': [has_photo],
        'has_description': [has_description],
        'description_length': [description_length],
        'title_length': [title_length],
        'square_feet': [square_feet],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'amenities_count': [amenities_count],
        'pets_allowed': [pets_allowed],
        'posting_month': [posting_month],
        'posting_day_of_week': [posting_day_of_week],
        'is_weekend': [is_weekend],
        'state_mean_price': [state_mean_price],
        'city_mean_price': [city_mean_price],
        'latitude': [latitude],
        'longitude': [longitude]
    })
    
    # Make prediction
    predicted_price = rf_model.predict(input_data)[0]
    
    print("\n" + "="*50)
    print("Listing Analysis Results")
    print("="*50)
    print(f"\nPredicted Rental Price: ${predicted_price:,.2f}")
    
    # Generate recommendations
    print("\nRecommendations to Optimize Your Listing:")
    
    if not has_photo:
        print("➤ Add photos to your listing - listings with photos tend to perform better")
    
    if description_length < df['description_length'].mean():
        print("➤ Consider adding more detail to your description")
        print(f"  Current length: {description_length} characters")
        print(f"  Average length: {int(df['description_length'].mean())} characters")
    
    if title_length < df['title_length'].mean():
        print("➤ Your title could be more descriptive")
        print(f"  Current length: {title_length} characters")
        print(f"  Average length: {int(df['title_length'].mean())} characters")
    
    if amenities_count < df['amenities_count'].mean():
        print("➤ Consider highlighting more amenities in your listing")
        print(f"  Current amenities: {amenities_count}")
        print(f"  Average amenities: {int(df['amenities_count'].mean())}")
    
    # Timing recommendations
    if posting_day_of_week != best_day:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"➤ Consider posting on {days[best_day]} for potentially better results")
    
    if posting_month != best_month:
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        print(f"➤ The best month for listings is typically {months[best_month-1]}")
    
    print("\nNote: These recommendations are based on historical data and market trends.")
    print("Individual results may vary based on local market conditions.")

except KeyboardInterrupt:
    print("\n\nAnalysis cancelled by user.")
except Exception as e:
    print(f"\n\nAn error occurred: {str(e)}") 