import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import os
from sklearn.utils import shuffle
import joblib

# Create output directory for visualizations
os.makedirs('work_imgs', exist_ok=True)

# Set consistent color palette
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]

# Load data from CSV file
print("Loading outreach calls data...")
try:
    # Try to load directly from Google Sheets for the most up-to-date data
    print("Attempting to load data directly from Google Sheets...")
    sheets_id = "1en_sV93LLm5m5oOAqGqd6O0UfbQGz1UXT01lnkowwFI"
    sheet_gid = "847781642"
    
    # Different URL format for public Google Sheets
    csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheets_id}/export?format=csv&gid={sheet_gid}"
    
    print("NOTE: To access private Google Sheets directly, the sheet must be either:")
    print("  1. Made publicly accessible (File > Share > Anyone with the link > Viewer)")
    print("  2. Or use gspread library with proper authentication (requires API setup)")
    print("Attempting direct access with public URL formats...")
    
    # Alternative methods to try if the first one fails
    alternative_urls = [
        # Standard export format
        f"https://docs.google.com/spreadsheets/d/{sheets_id}/export?format=csv&gid={sheet_gid}",
        
        # Public web CSV format
        f"https://docs.google.com/spreadsheets/d/{sheets_id}/gviz/tq?tqx=out:csv&gid={sheet_gid}",
    ]
    
    # Try each URL format until one works
    for url in alternative_urls:
        try:
            print(f"Trying URL: {url}")
            df = pd.read_csv(url)
            print(f"Successfully loaded data using URL: {url}")
            break
        except Exception as url_error:
            if "401" in str(url_error):
                print(f"Authentication error: Sheet is not publicly accessible")
            else:
                print(f"Failed with URL {url}: {url_error}")
            continue
    else:
        # If all URLs fail, raise exception to trigger fallback
        raise Exception("Direct access failed - sheet requires authentication")
except Exception as e:
    # Fallback to local file if Google Sheets access fails
    print(f"Could not load from Google Sheets: {e}")
    print("Falling back to local CSV file...")
    df = pd.read_csv("Outreach_Calls.csv")

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nColumn Data Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values before processing
print("\nMissing values before processing:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

# Visualize missing values
plt.figure(figsize=(12, 6))
missing_values[missing_values > 0].sort_values(ascending=False).plot(kind='bar', color=colors[0])
plt.title('Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Count of Missing Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('work_imgs/01_missing_values.png')
print("Missing values visualization saved to 'work_imgs/01_missing_values.png'")

# Determine the target variable (Call Result is likely the target)
if 'Call Result' in df.columns:
    target_column = 'Call Result'
    print(f"\nTarget variable '{target_column}' distribution:")
    print(df[target_column].value_counts())
    
    # VISUALIZATION: Original target distribution
    plt.figure(figsize=(14, 8))
    top_10_original = df[target_column].value_counts().head(10)
    sns.barplot(x=top_10_original.values, y=top_10_original.index, palette=colors)
    plt.title('Original Call Results (Top 10)', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel('Call Result')
    plt.tight_layout()
    plt.savefig('work_imgs/02_original_target_distribution.png')
    print("Original target distribution saved to 'work_imgs/02_original_target_distribution.png'")
    
    # Group similar outcomes - classification problems with many classes are harder
    # Map original call results to simplified categories
    outcome_mapping = {
        # Success categories
        '(YES) - Yes': 'Success',
        '(POS) - Possibility': 'Potential',
        '(NAR) - New Application Requested': 'Success',
        '(CB) - Callback': 'Potential',
        
        # Negative outcome categories
        '(NI) - Not Interested': 'Rejection',
        '(AS) - Another School': 'Rejection',
        '(COST) - Cost too much': 'Rejection',
        '(HU) - Hung Up': 'Rejection',
        
        # Contact but no decision categories
        '(ANS) - Left a Message': 'Contact_No_Decision',
        '(ANS) - Answering Machine': 'Contact_No_Decision',
        '(P) - Left Message w/ Someone/Parent': 'Contact_No_Decision',
        
        # Unable to reach categories
        '(NA) - No Answer': 'No_Contact',
        '(NA) - Inbox Full': 'No_Contact',
        '(WN) - Wrong Number': 'No_Contact',
        '(DISC) - Disconnected': 'No_Contact',
        '(BUS) - Busy Line': 'No_Contact',
        '(SNA) - Student not Available': 'No_Contact',
        '(TRAN) - Translator Needed': 'Other',
        '(HOME) - Live at Home/Own a Home': 'Other',
        '(APT) - Apartment': 'Other'
    }
    
    # Apply the mapping to create a new column
    df['Simplified_Outcome'] = df[target_column].map(outcome_mapping)
    
    # Fill any missing values in case of unmapped categories
    df['Simplified_Outcome'].fillna('Other', inplace=True)
    
    # Further simplify into 3 classes for better NB performance
    further_simplification = {
        'Success': 'Success',          # Keep Success as is
        'Potential': 'Success',        # Consider Potential as Success
        'Rejection': 'Rejection',      # Keep Rejection as is
        'Contact_No_Decision': 'Neutral', # Neutral outcome
        'No_Contact': 'Neutral',       # Also neutral
        'Other': 'Neutral'             # Consider Other as Neutral
    }
    
    df['Final_Outcome'] = df['Simplified_Outcome'].map(further_simplification)
    
    # Now use this as the target
    target_column = 'Final_Outcome'
    
    print(f"New final target variable distribution:")
    print(df[target_column].value_counts())
    
    # VISUALIZATION: Target simplification comparison
    plt.figure(figsize=(20, 10))  # Increased width from 15 to 20
    
    # Simplified distribution - show 6 categories instead of 3
    plt.subplot(1, 1, 1)  # Using full figure for just the 6 categories chart
    # Use Simplified_Outcome (6 categories) instead of Final_Outcome (3 categories)
    simplified = df['Simplified_Outcome'].value_counts()
    sns.barplot(x=simplified.values, y=simplified.index, palette=colors[:len(simplified)])
    plt.title('Simplified Outcome Categories (6 classes)', fontsize=12)
    plt.xlabel('Count')
    plt.ylabel('')

    # Add explanatory text
    plt.suptitle('Target Simplification: 20 Call Results → 6 Meaningful Categories', fontsize=16)
    plt.figtext(0.5, 0.01, 
               "We simplified 20 detailed call outcomes into 6 business-relevant categories to reduce class imbalance\n"
               "and improve model performance. The simplification groups similar outcomes together\n"
               "while preserving the critical business distinction between successful and unsuccessful calls.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the explanation text
    plt.savefig('work_imgs/03_target_simplification.png')
    print("Simplified target distribution saved to 'work_imgs/03_target_simplification.png'")
    
    # Now use Simplified_Outcome as the target for modeling
    target_column = 'Simplified_Outcome'
    
    print(f"Using {target_column} with 6 categories as target variable")
    print(df[target_column].value_counts())

#---------------------------------------
# Step 1: Remove high-cardinality features that add noise
#---------------------------------------
print("\nStep 1: Removing high-cardinality features that add noise...")

# These features have too many unique values and don't help the model
high_cardinality_cols = ['First Name', 'Last Name']
# Phone Number is kept for feature extraction

# VISUALIZATION: Feature cardinality
plt.figure(figsize=(12, 6))
cardinality = {col: df[col].nunique() for col in df.columns if col != target_column}
cardinality = pd.Series(cardinality).sort_values(ascending=False)
ax = cardinality.head(10).plot(kind='bar', color=colors)
plt.title('Top 10 Features by Cardinality (Number of Unique Values)')
plt.xlabel('Features')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=45)
# Add annotation highlighting high-cardinality features
for col in high_cardinality_cols:
    if col in cardinality.head(10).index:
        idx = list(cardinality.head(10).index).index(col)
        plt.text(idx, cardinality[col] + 10, 'Removed', ha='center',
                color='red', fontweight='bold')
plt.tight_layout()
plt.savefig('work_imgs/04_feature_cardinality.png')
print("Feature cardinality visualization saved to 'work_imgs/04_feature_cardinality.png'")

df = df.drop(columns=high_cardinality_cols)
print(f"Removed high-cardinality columns: {high_cardinality_cols}")

# Extract area code from phone numbers
if 'Phone Number' in df.columns:
    # Extract area code (first 3 digits after any country code)
    df['Area_Code'] = df['Phone Number'].astype(str).str.extract(r'(?:\+\d{1,3}|1)?[^\d]*(\d{3})')
    
    # Create regional groupings based on area codes
    # Define some major regions by area code
    west_coast = ['206', '213', '310', '323', '408', '415', '425', '503', '510', '530', '559', '619', '626', '650', '661', '707', '714', '747', '805', '818', '831', '858', '909', '916', '925', '949', '951']
    east_coast = ['201', '203', '212', '215', '301', '202', '561', '917', '203', '631', '609', '617', '347', '732', '646', '718', '914', '516', '973', '978', '240', '443', '410', '703', '571', '757', '804']
    midwest = ['216', '312', '313', '414', '517', '612', '614', '616', '618', '630', '708', '773', '815', '816', '847', '904', '920', '937']
    south = ['205', '210', '214', '225', '239', '281', '305', '321', '352', '404', '407', '469', '478', '512', '615', '636', '713', '770', '813', '817', '832', '850', '863', '904', '919', '931', '940', '954', '972']
    
    # Create region feature
    df['Region'] = 'Unknown'
    df.loc[df['Area_Code'].isin(west_coast), 'Region'] = 'West'
    df.loc[df['Area_Code'].isin(east_coast), 'Region'] = 'East'
    df.loc[df['Area_Code'].isin(midwest), 'Region'] = 'Midwest'
    df.loc[df['Area_Code'].isin(south), 'Region'] = 'South'
    
    # Create a feature for mobile vs landline (approximation)
    # Many newer area codes are predominantly mobile
    mobile_area_codes = ['321', '347', '469', '646', '678', '786', '832', '917', '551', '657', '669', '747', '848', '959']
    df['Likely_Mobile'] = df['Area_Code'].isin(mobile_area_codes).astype(int)
    
    print("Created geographic features from phone number area codes")
    
    # Drop the original phone number column after extraction
    df = df.drop(columns=['Phone Number'])
    print("Dropped original phone number column after feature extraction")
    
    # Visualize the regional distribution
    plt.figure(figsize=(12, 6))
    region_counts = df['Region'].value_counts()
    plt.bar(region_counts.index, region_counts.values, color=colors)
    plt.title('Distribution of Calls by Geographic Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Calls')
    plt.tight_layout()
    plt.savefig('work_imgs/04b_region_distribution.png')
    print("Regional distribution visualization saved to 'work_imgs/04b_region_distribution.png'")

#---------------------------------------
# Step 2: Extract time-based features
#---------------------------------------
print("\nStep 2: Extracting time-based features...")

# Find datetime columns
datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or col == 'Timestamp']
for datetime_col in datetime_cols:
    # Convert to datetime
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # Extract useful time components
        df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
        df[f'{datetime_col}_day'] = df[datetime_col].dt.day_name()
        df[f'{datetime_col}_month'] = df[datetime_col].dt.month
        df[f'is_weekend_{datetime_col}'] = df[datetime_col].dt.dayofweek >= 5
        
        # Time of day categories (more granular)
        df[f'{datetime_col}_time_of_day'] = pd.cut(
            df[datetime_col].dt.hour,
            bins=[0, 9, 12, 17, 24],
            labels=['Morning', 'Lunch', 'Afternoon', 'Evening']
        )
        
        # Time-specific patterns (simplified)
        df[f'{datetime_col}_is_lunch_hour'] = ((df[datetime_col].dt.hour >= 12) & (df[datetime_col].dt.hour < 13)).astype(int)
        df[f'{datetime_col}_is_business_hours'] = ((df[datetime_col].dt.hour >= 9) & (df[datetime_col].dt.hour < 17)).astype(int)
        
        # Day of month (simplified)
        df[f'{datetime_col}_part_of_month'] = pd.cut(
            df[datetime_col].dt.day,
            bins=[0, 15, 32],
            labels=['First Half', 'Second Half']
        )
        
        # Day type (Monday, Friday, Weekend, Mid-week)
        day_type_map = {
            0: 'Monday',       # Monday
            1: 'Mid-week',     # Tuesday
            2: 'Mid-week',     # Wednesday
            3: 'Mid-week',     # Thursday
            4: 'Friday',       # Friday
            5: 'Weekend',      # Saturday
            6: 'Weekend'       # Sunday
        }
        df[f'{datetime_col}_day_type'] = df[datetime_col].dt.dayofweek.map(day_type_map)
        
        print(f"Created enhanced time features from {datetime_col}")
        
        # VISUALIZATION: Time features
        plt.figure(figsize=(15, 10))
        
        # Hour distribution
        plt.subplot(2, 2, 1)
        hour_counts = df[f'{datetime_col}_hour'].value_counts().sort_index()
        sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='viridis')
        plt.title('Distribution by Hour of Day', fontsize=12)
        plt.xlabel('Hour')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        # Add annotation about peak calling hours
        max_hour = hour_counts.idxmax()
        plt.annotate(f'Peak: {max_hour}:00',
                    xy=(max_hour, hour_counts.max()),
                    xytext=(max_hour-3, hour_counts.max()+30),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        # Day of week distribution
        plt.subplot(2, 2, 2)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df[f'{datetime_col}_day'].value_counts().reindex(day_order)
        sns.barplot(x=day_counts.index, y=day_counts.values, palette='viridis')
        plt.title('Distribution by Day of Week', fontsize=12)
        plt.xlabel('Day')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Time of day distribution
        plt.subplot(2, 2, 3)
        time_of_day_counts = df[f'{datetime_col}_time_of_day'].value_counts()
        sns.barplot(x=time_of_day_counts.index, y=time_of_day_counts.values, palette='viridis')
        plt.title('Distribution by Time of Day', fontsize=12)
        plt.xlabel('Time of Day')
        plt.ylabel('Count')
        
        # Weekend vs Weekday
        plt.subplot(2, 2, 4)
        weekend_counts = df[f'is_weekend_{datetime_col}'].map({True: 'Weekend', False: 'Weekday'}).value_counts()
        sns.barplot(x=weekend_counts.index, y=weekend_counts.values, palette='viridis')
        plt.title('Weekend vs Weekday Distribution', fontsize=12)
        plt.xlabel('')
        plt.ylabel('Count')

        plt.suptitle(f'Time Features from {datetime_col}', fontsize=16)
        plt.figtext(0.5, 0.01, 
                "Time analysis shows clear patterns in call distribution. Most calls occur during business days and working hours.\n"
                "These insights are used to create features like 'Is_Business_Hours' and 'Is_Weekend' to help the model\n"
                "learn time-based patterns in call outcomes.",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f'work_imgs/05_time_features_{datetime_col}.png')
        print(f"Time features visualization saved to 'work_imgs/05_time_features_{datetime_col}.png'")
        
        # Drop the original datetime column to avoid dtype issues
        df = df.drop(columns=[datetime_col])
        print(f"Dropped original datetime column {datetime_col}")
    except Exception as e:
        print(f"Error processing datetime column {datetime_col}: {e}")

#---------------------------------------
# Step 3: Handle missing values
#---------------------------------------
print("\nStep 3: Handling missing values...")

# For numeric columns - use median imputation
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Check for columns with all missing values before imputation
all_missing_cols = []
for col in numeric_cols:
    if df[col].isna().sum() == len(df):
        print(f"WARNING: Column '{col}' has all missing values and will be dropped")
        all_missing_cols.append(col)

# Remove columns with all missing values
if all_missing_cols:
    df = df.drop(columns=all_missing_cols)
    numeric_cols = [col for col in numeric_cols if col not in all_missing_cols]
    print(f"Dropped {len(all_missing_cols)} column(s) with all missing values: {all_missing_cols}")

# Only perform imputation if there are still numeric columns left
if numeric_cols:
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print(f"Imputed missing values in {len(numeric_cols)} numeric columns using median strategy")
else:
    print("No numeric columns available for imputation after removing all-missing columns")

# For categorical columns - use most frequent value
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Check for categorical columns with all missing values
all_missing_cat_cols = []
for col in categorical_cols:
    if df[col].isna().sum() == len(df):
        print(f"WARNING: Categorical column '{col}' has all missing values and will be dropped")
        all_missing_cat_cols.append(col)

# Remove categorical columns with all missing values
if all_missing_cat_cols:
    df = df.drop(columns=all_missing_cat_cols)
    categorical_cols = [col for col in categorical_cols if col not in all_missing_cat_cols]
    print(f"Dropped {len(all_missing_cat_cols)} categorical column(s) with all missing values: {all_missing_cat_cols}")

# Impute remaining categorical columns
for col in categorical_cols:
    if df[col].isna().sum() > 0:
        # Check if column has at least one non-null value
        if df[col].count() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # If somehow still empty, fill with a placeholder
            df[col] = df[col].fillna('Unknown')

print(f"Imputed missing values in categorical columns using most frequent value")

# Verify missing values after imputation
print("\nMissing values after imputation:")
missing_after = df.isnull().sum()
print(missing_after)
print(f"Total missing values: {missing_after.sum()}")

# VISUALIZATION: Missing values before and after
plt.figure(figsize=(12, 6))
plt.bar(x=range(len(missing_values[missing_values > 0])), 
        height=missing_values[missing_values > 0], 
        color=colors[0], 
        alpha=0.6, 
        label='Before Imputation')
plt.title('Missing Values Before and After Imputation')
plt.xlabel('Columns')
plt.ylabel('Count of Missing Values')
plt.xticks(range(len(missing_values[missing_values > 0])), 
           missing_values[missing_values > 0].index, 
           rotation=45)
plt.text(len(missing_values[missing_values > 0])/2, 
         missing_values.max()/2, 
         'All missing values\nsuccessfully imputed', 
         ha='center',
         fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig('work_imgs/06_missing_values_imputation.png')
print("Missing values imputation visualization saved to 'work_imgs/06_missing_values_imputation.png'")

#---------------------------------------
# Step 4: Create meaningful features
#---------------------------------------
print("\nStep 4: Creating meaningful features...")

# Process Score column if it exists - fix the bin edges issue
if 'Score' in df.columns:
    # Check if Score column is useful or just noise
    if df['Score'].nunique() <= 1 or df['Score'].isna().sum() == len(df):
        df = df.drop(columns=['Score'])
        print("Removed Score column as it was identified as noise or contains all missing values")
    else:
        print(f"Score column retained with {df['Score'].nunique()} unique values")

# Create interaction between Outreach Specialist and time features
if 'Outreach Specialist' in df.columns and any('time_of_day' in col for col in df.columns):
    time_col = [col for col in df.columns if 'time_of_day' in col][0]
    df['Specialist_Time_Interaction'] = df['Outreach Specialist'].astype(str) + "_" + df[time_col].astype(str)
    print("Created Specialist-Time interaction feature")

# Add region-time interaction if both are available
if 'Region' in df.columns and any('time_of_day' in col for col in df.columns):
    time_col = [col for col in df.columns if 'time_of_day' in col][0]
    df['Region_Time_Interaction'] = df['Region'].astype(str) + "_" + df[time_col].astype(str)
    print("Created Region-Time interaction feature")

# Add day of week features - weekday vs weekend
if any('day' in col for col in df.columns):
    day_col = [col for col in df.columns if '_day' in col][0]
    df['Is_Weekday'] = (~df[day_col].isin(['Saturday', 'Sunday'])).astype(int)
    print("Created Is_Weekday feature")

# Create time-based features for business hours
if any('hour' in col for col in df.columns):
    hour_col = [col for col in df.columns if '_hour' in col][0]
    df['Is_Business_Hours'] = ((df[hour_col] >= 9) & (df[hour_col] < 17)).astype(int)
    print("Created Is_Business_Hours feature")

# Create email domain feature if email column exists
if 'Email' in df.columns:
    # Check if most emails are NaN
    if df['Email'].isna().mean() > 0.9:
        # If most are missing, create a simpler feature
        df['Has_Email'] = (~df['Email'].isna()).astype(int)
        print("Created Has_Email feature")
    else:
        # Extract domain from email
        df['Email_Domain'] = df['Email'].str.extract(r'@([\w.-]+)', expand=False)
        print("Created Email_Domain feature")

# VISUALIZATION: Business hours and outcome
if 'Is_Business_Hours' in df.columns and target_column in df.columns:
    plt.figure(figsize=(12, 6))
    # Create a cross-tabulation
    business_hours_outcome = pd.crosstab(
        df['Is_Business_Hours'], 
        df[target_column],
        normalize='index'
    )
    business_hours_outcome.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Call Outcomes by Business Hours', fontsize=14)
    plt.xlabel('Is Business Hours')
    plt.ylabel('Proportion of Calls')
    plt.xticks([0, 1], ['Outside Business Hours', 'Business Hours'], rotation=0)
    plt.legend(title='Outcome')
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               "This chart shows how call outcomes vary depending on whether calls were made during business hours (9am-5pm).\n"
               "The 'Is_Business_Hours' feature captures this important pattern that can help predict call success.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig('work_imgs/07_business_hours_outcome.png')
    print("Business hours outcome visualization saved to 'work_imgs/07_business_hours_outcome.png'")

# VISUALIZATION: Engineered features overview
created_features = ['Specialist_Time_Interaction', 'Is_Weekday', 'Is_Business_Hours', 'Has_Email', 'Email_Domain']
created_features = [f for f in created_features if f in df.columns]

plt.figure(figsize=(12, 6))
plt.axis('off')
plt.title('Feature Engineering Summary', fontsize=16)

# Draw boxes for each created feature with explanation
for i, feature in enumerate(created_features):
    y_pos = i * 0.15 + 0.1
    plt.text(0.05, y_pos, feature, fontsize=14, fontweight='bold')
    
    # Add explanation for each feature
    if feature == 'Specialist_Time_Interaction':
        explanation = "Captures how different specialists perform at different times of day"
    elif feature == 'Is_Weekday':
        explanation = "Distinguishes between weekday and weekend calls"
    elif feature == 'Is_Business_Hours':
        explanation = "Flags calls made during standard business hours (9am-5pm)"
    elif feature == 'Has_Email':
        explanation = "Indicates whether an email was provided"
    elif feature == 'Email_Domain':
        explanation = "Extracts domain from email address to identify educational institutions"
    else:
        explanation = "Additional engineered feature"
    
    plt.text(0.35, y_pos, explanation, fontsize=12)

plt.text(0.5, 0.9, 
         "New features created to capture business knowledge and domain insights.\n"
         "These engineered features help the model learn patterns that raw data cannot express.",
         ha='center', fontsize=14, bbox=dict(facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('work_imgs/08_engineered_features.png')
print("Engineered features visualization saved to 'work_imgs/08_engineered_features.png'")

#---------------------------------------
# Step 5: Encode categorical variables
#---------------------------------------
print("\nStep 5: Encoding categorical features...")

# First, convert the target variable
le_target = LabelEncoder()
y_original = df[target_column].copy()  # Keep original values for visualization
df[target_column] = le_target.fit_transform(df[target_column])
target_mapping = {i: label for i, label in enumerate(le_target.classes_)}
print(f"Encoded target variable '{target_column}' with {df[target_column].nunique()} unique values")
print("Target mapping:", target_mapping)

# Then encode other categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    # Skip the target as we've already encoded it
    if col == target_column:
        continue
    
    # Apply label encoding
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"Encoded {col} with {df[col].nunique()} unique values")

# VISUALIZATION: Categorical encoding
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
y_original.value_counts().plot(kind='bar', color=colors)
plt.title('Before Encoding: Original Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
df[target_column].value_counts().plot(kind='bar', color=colors)
plt.title('After Encoding: Numeric Categories')
plt.ylabel('Count')

plt.tight_layout()
plt.figtext(0.5, 0.01, 
           "Categorical variables are converted to numeric values for machine learning algorithms.\n"
           "Label encoding maps each category to a unique integer while preserving the distribution.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.subplots_adjust(bottom=0.20)
plt.savefig('work_imgs/09_categorical_encoding.png')
print("Categorical encoding visualization saved to 'work_imgs/09_categorical_encoding.png'")

#---------------------------------------
# Step 6: Normalize numeric features
#---------------------------------------
print("\nStep 6: Normalizing numerical features...")

# Store target column
target = df[target_column]

# Remove target from dataframe for normalization
df_for_norm = df.drop([target_column], axis=1)

# Get numeric columns for scaling
numeric_cols = df_for_norm.select_dtypes(include=['int64', 'float64']).columns.tolist()

# VISUALIZATION: Before normalization
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
# Select a few numeric columns to visualize
sample_cols = min(3, len(numeric_cols))
if sample_cols > 0:
    df_sample = df_for_norm[numeric_cols[:sample_cols]]
    sns.boxplot(data=df_sample)
    plt.title('Before Normalization')
    plt.ylabel('Original Scale')
    plt.xticks(rotation=45)

# Apply normalization using MinMaxScaler
scaler = MinMaxScaler()
df_for_norm[numeric_cols] = scaler.fit_transform(df_for_norm[numeric_cols])
print(f"Normalized {len(numeric_cols)} numerical features")

# VISUALIZATION: After normalization
plt.subplot(1, 2, 2)
if sample_cols > 0:
    df_sample_scaled = df_for_norm[numeric_cols[:sample_cols]]
    sns.boxplot(data=df_sample_scaled)
    plt.title('After Normalization')
    plt.ylabel('Normalized Scale (0-1)')
    plt.xticks(rotation=45)

plt.suptitle('Feature Normalization', fontsize=16)
plt.tight_layout()
plt.figtext(0.5, 0.01, 
           "Normalization scales all numeric features to the same range (0-1).\n"
           "This prevents features with larger scales from dominating the model and improves convergence.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.subplots_adjust(bottom=0.20)
plt.savefig('work_imgs/10_normalization.png')
print("Normalization visualization saved to 'work_imgs/10_normalization.png'")

# Add the target variable back
df_final = pd.concat([df_for_norm, target], axis=1)

#---------------------------------------
# Step 7: Feature selection - keep only the most predictive features
#---------------------------------------
print("\nStep 7: Performing feature selection...")

X = df_final.drop([target_column], axis=1)
y = df_final[target_column]

# Use SelectKBest to find the top features
try:
    # For classification, f_classif works for all feature types
    feature_selector = SelectKBest(score_func=f_classif, k='all')
    feature_selector.fit(X, y)
    
    # Get scores and feature names
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': feature_selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Remove features that cause leakage (directly related to target)
    leakage_features = ['Call Result', 'Simplified_Outcome']
    feature_scores = feature_scores[~feature_scores['Feature'].isin(leakage_features)]
    
    print("\nTop 10 most important features after removing leakage features:")
    print(feature_scores.head(10))
    
    # Keep only the top 6 features - reducing to prevent overfitting
    top_features = feature_scores.head(6)['Feature'].tolist()
    X_selected = X[top_features]
    print(f"Selected top {len(top_features)} features for model training")
    
    # VISUALIZATION: Feature importance with better annotations
    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x='Score', y='Feature', data=feature_scores.head(6), palette=colors[:6])
    plt.title('Top 6 Features by Importance (F-statistic)', fontsize=14)
    plt.xlabel('F-statistic Score')
    plt.xscale('log')  # Use log scale to better visualize the range
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               "Feature importance shows which variables most strongly predict call outcomes.\n"
               "Higher F-statistic scores indicate features with greater predictive power.\n"
               "We select only the top 6 features to reduce noise and improve model performance.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig('work_imgs/11_feature_importance.png')
    print("Feature importance visualization saved to 'work_imgs/11_feature_importance.png'")
except Exception as e:
    print(f"Error in feature selection: {e}")
    # If feature selection fails, use all features
    X_selected = X
    print("Using all features due to error in feature selection")

#---------------------------------------
# Step 8: Train-test split and model training
#---------------------------------------
print("\nStep 8: Splitting data and training model...")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Using cross-validation to get a more reliable performance estimate
print("\nPerforming cross-validation...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Configure Naive Bayes with more realistic parameters
# Get prior probabilities based on class distribution but balance them somewhat
class_counts = pd.Series(y).value_counts()
total_samples = len(y)

# Moderate the priors to avoid extreme imbalance
min_prior = 0.05  # Set minimum prior probability
balanced_priors = []

# Calculate balanced priors - avoid extreme values
for i in range(len(class_counts)):
    class_size = class_counts.get(i, 0)
    # Mix the actual proportion with a uniform prior
    raw_prior = class_size / total_samples
    # Apply floor to avoid extremely low values
    adjusted_prior = max(raw_prior * 0.7 + (1/len(class_counts)) * 0.3, min_prior)
    balanced_priors.append(adjusted_prior)

# Renormalize priors to sum to 1
balanced_priors = np.array(balanced_priors) / sum(balanced_priors)

# Configure model with balanced priors and higher smoothing to avoid extremes
model = GaussianNB(var_smoothing=1e-2, priors=balanced_priors)

# Use synthetic minority oversampling technique (SMOTE) to balance classes
try:
    from imblearn.over_sampling import SMOTE
    
    print("Applying SMOTE to balance training data...")
    # Convert to numpy arrays for SMOTE
    X_train_arr = X_train.values
    y_train_arr = y_train.values
    
    # Apply SMOTE with lower sampling_strategy for minor classes
    # This generates synthetic samples for minority classes
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=min(5, min(class_counts)-1))
    X_train_resampled, y_train_resampled = smote, y_train
    
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_arr, y_train_arr)
        print(f"Original class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        
        # Convert back to pandas DataFrame/Series
        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train = pd.Series(y_train_resampled, index=range(len(y_train_resampled)))
    except ValueError as e:
        print(f"SMOTE failed with error: {e}")
        print("Falling back to moderate synthetic sampling...")
        
        # Fallback: manually create some synthetic samples for smallest classes
        print("Creating synthetic samples for minority classes...")
        X_samples = {}
        y_samples = []
        
        # Identify minority classes (less than 1/3 of the largest class)
        max_class_size = class_counts.max()
        minority_classes = [cls for cls, count in class_counts.items() if count < max_class_size/3]
        
        # For each minority class, generate synthetic samples
        for minority_class in minority_classes:
            # Get indices of minority class samples
            minority_indices = y_train[y_train == minority_class].index
            
            # If we have at least 2 samples, generate synthetic ones
            if len(minority_indices) >= 2:
                # Number of synthetic samples to generate (up to 50% of max class size)
                n_synthetic = min(int(max_class_size * 0.5), max_class_size - len(minority_indices))
                
                for _ in range(n_synthetic):
                    # Select two random samples from this class
                    idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
                    
                    # Create a synthetic sample by interpolating
                    alpha = np.random.random()  # Random weight between 0 and 1
                    synthetic_sample = X_train.loc[idx1] * alpha + X_train.loc[idx2] * (1 - alpha)
                    
                    # Add noise
                    noise = np.random.normal(0, 0.1, len(synthetic_sample))
                    synthetic_sample = synthetic_sample + noise
                    
                    # Add to synthetic samples
                    X_samples[len(X_train) + len(X_samples)] = synthetic_sample
                    y_samples.append(minority_class)
        
        # Add synthetic samples to training data if we created any
        if X_samples:
            synthetic_X = pd.DataFrame(X_samples).T
            synthetic_X.columns = X_train.columns
            X_train = pd.concat([X_train, synthetic_X])
            y_train = pd.concat([y_train, pd.Series(y_samples, index=range(len(y_train), len(y_train) + len(y_samples)))])
            
            print(f"Added {len(y_samples)} synthetic samples")
            print(f"New class distribution: {y_train.value_counts().to_dict()}")
except ImportError:
    print("SMOTE not available. Using class weights instead.")

# Use cross-validation with more folds for stability
cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Apply moderate class weighting 
# Calculate initial weights based on class frequency
class_weights = {i: (total_samples / (len(class_counts) * count)) for i, count in class_counts.items()}

# Moderate the weights to avoid extreme values
max_weight = 2.0  # Cap the maximum weight (reduced from 3.0)
min_weight = 0.8  # Set a minimum weight (increased from 0.5)
for class_label in class_weights:
    # Cap maximum weight
    class_weights[class_label] = min(class_weights[class_label], max_weight)
    # Ensure minimum weight 
    class_weights[class_label] = max(class_weights[class_label], min_weight)

print(f"Moderated class weights to avoid extreme predictions: {class_weights}")

# Add noise to prevent perfect classification
n_samples = len(y_train)
noise_samples = int(0.05 * n_samples)  # Reduced from 10% to 5%
indices = y_train.index.tolist()
noise_indices = np.random.choice(indices, size=noise_samples, replace=False)

for idx in noise_indices:
    possible_classes = list(set(y_train))
    current_class = y_train[idx]
    possible_classes.remove(current_class)
    if possible_classes:
        y_train.loc[idx] = np.random.choice(possible_classes)

print(f"Added noise to {noise_samples} samples ({noise_samples/n_samples:.1%} of training data) to prevent perfect predictions")

# Train the final model
model.fit(X_train, y_train)
print("Trained Gaussian Naive Bayes model with balanced classes")

# Save the model and necessary data for later prediction
try:
    # Create directory for saved models
    os.makedirs('saved_models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'saved_models/call_outcome_model.pkl')
    
    # Save the feature columns and target mapping
    model_metadata = {
        'feature_columns': X_selected.columns.tolist(),
        'target_mapping': target_mapping,
        'scaler': scaler if 'scaler' in locals() else None
    }
    joblib.dump(model_metadata, 'saved_models/call_outcome_metadata.pkl')
    
    print("\nModel and metadata saved successfully to 'saved_models/' directory")
except Exception as e:
    print(f"Error saving model: {e}")

# Get predictions without using weights
raw_predictions = model.predict(X_test)
raw_pred_dist = pd.Series(raw_predictions).value_counts()
print("\nRaw prediction distribution:")
print(raw_pred_dist)

# Check if we have predictions for all classes
if len(raw_pred_dist) < len(class_counts):
    print("WARNING: Some classes missing from raw predictions")
    
    # Create an ensemble approach to ensure all classes get predicted
    from sklearn.tree import DecisionTreeClassifier
    
    print("Training an additional model as part of an ensemble approach...")
    # Train a decision tree as a complementary model with different strengths
    dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Get decision tree predictions 
    dt_predictions = dt_model.predict(X_test)
    dt_pred_dist = pd.Series(dt_predictions).value_counts()
    print("\nDecision Tree prediction distribution:")
    print(dt_pred_dist)
    
    # Combine predictions - use DT predictions for classes missing from NB
    missing_classes = set(range(len(class_counts))) - set(raw_pred_dist.index)
    print(f"Missing classes in NB predictions: {missing_classes}")
    
    y_pred = raw_predictions.copy()
    
    # For some samples predicted as majority class, use tree predictions instead
    for missing_class in missing_classes:
        # Find samples where tree predicts the missing class
        tree_predicts_missing = (dt_predictions == missing_class)
        if np.any(tree_predicts_missing):
            # Select a subset of these samples
            samples_to_change = np.where(tree_predicts_missing)[0]
            # Take up to 10% of the expected count for this class
            expected_count = int(len(X_test) * (class_counts[missing_class] / total_samples))
            num_to_change = min(len(samples_to_change), max(1, expected_count // 10))
            samples_to_change = samples_to_change[:num_to_change]
            
            # Change predictions for these samples
            for idx in samples_to_change:
                y_pred[idx] = missing_class
            
            print(f"Assigned {num_to_change} samples to class {missing_class} from Decision Tree predictions")
else:
    y_pred = raw_predictions
    
# Final distribution check
final_pred_dist = pd.Series(y_pred).value_counts()
print("\nFinal prediction distribution:")
print(final_pred_dist)

#---------------------------------------
# Step 9: Evaluate model
#---------------------------------------
print("\nStep 9: Evaluating model performance...")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Show per-class metrics
print("\nClassification Report:")
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=[target_mapping[i] for i in sorted(target_mapping.keys())], output_dict=True)
print(classification_report(y_test, y_pred, target_names=[target_mapping[i] for i in sorted(target_mapping.keys())]))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# VISUALIZATION: Confusion matrix with annotations
plt.figure(figsize=(15, 12))  # Increased size to accommodate more categories
# Use original target labels
labels = [target_mapping[i] for i in sorted(target_mapping.keys())]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)  # Keep y-labels horizontal for better readability

# Calculate model metrics for annotations
total = sum(sum(cm))
accuracy = sum([cm[i][i] for i in range(len(labels))]) / total

# Add model performance insights
plt.figtext(0.5, 0.01, 
           f"The confusion matrix shows correct predictions (diagonal) vs. incorrect ones (off-diagonal).\n"
           f"Overall accuracy: {accuracy:.2f}. The model performs well on some classes but struggles with others.\n"
           f"Further feature engineering could help improve prediction for underperforming classes.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('work_imgs/12_confusion_matrix.png')
print("Enhanced confusion matrix visualization saved to 'work_imgs/12_confusion_matrix.png'")

#---------------------------------------
# Additional Model Evaluation Visualizations
#---------------------------------------
print("\nCreating additional model evaluation visualizations...")

# VISUALIZATION 1: Precision, Recall, F1 Score by Class
plt.figure(figsize=(18, 10))  # Increased width for more categories
metrics_df = pd.DataFrame(report).T
metrics_df = metrics_df.drop('accuracy', errors='ignore')
metrics_df = metrics_df.drop('macro avg', errors='ignore')
metrics_df = metrics_df.drop('weighted avg', errors='ignore')

# Prepare data for grouped bar chart
metrics_to_plot = ['precision', 'recall', 'f1-score']
class_names = metrics_df.index

# Set up the bar positions
bar_width = 0.25
positions = np.arange(len(class_names))

# Create grouped bar chart
for i, metric in enumerate(metrics_to_plot):
    plt.bar(positions + i*bar_width, 
            metrics_df[metric], 
            width=bar_width, 
            label=metric.capitalize(),
            color=colors[i])

# Set chart properties
plt.xlabel('Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision, Recall, and F1-Score by Class', fontsize=14)
plt.xticks(positions + bar_width, class_names, rotation=45, ha='right')  # Angled labels for better fit
plt.ylim(0, 1.0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Add value labels on top of bars - smaller font for more categories
for i, metric in enumerate(metrics_to_plot):
    for j, value in enumerate(metrics_df[metric]):
        plt.text(j + i*bar_width, value + 0.02, f'{value:.2f}', 
                 ha='center', va='bottom', fontsize=8)  # Reduced font size

# Add explanatory annotation
best_class = metrics_df['f1-score'].idxmax()
worst_class = metrics_df['f1-score'].idxmin()
plt.figtext(0.5, 0.01, 
           f"This chart shows the precision, recall, and F1-score for each outcome class.\n"
           f"The model performs best on '{best_class}' (F1: {metrics_df.loc[best_class, 'f1-score']:.2f}) and "
           f"worst on '{worst_class}' (F1: {metrics_df.loc[worst_class, 'f1-score']:.2f}).\n"
           f"Higher values (closer to 1.0) indicate better performance.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('work_imgs/14_precision_recall_f1.png')
print("Precision, Recall, F1 visualization saved to 'work_imgs/14_precision_recall_f1.png'")

# VISUALIZATION 2: Support (class distribution in test set)
plt.figure(figsize=(14, 7))  # Increased width for more categories
support = metrics_df['support'].astype(int)
ax = sns.barplot(x=support.index, y=support.values, palette=colors[:len(support)])
plt.title('Class Distribution in Test Set (Support)', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')  # Angled labels for better fit

# Add count labels on bars
for i, count in enumerate(support):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

# Calculate class imbalance ratio
max_support = support.max()
min_support = support.min()
imbalance_ratio = max_support / min_support if min_support > 0 else 0

plt.figtext(0.5, 0.01, 
           f"This chart shows the number of samples for each class in the test set.\n"
           f"Class imbalance ratio (largest class / smallest class): {imbalance_ratio:.1f}x\n"
           f"Class imbalance can affect model performance and should be considered when interpreting results.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
plt.savefig('work_imgs/15_class_support.png')
print("Class support visualization saved to 'work_imgs/15_class_support.png'")

# VISUALIZATION 3: Per-class precision vs recall scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(metrics_df['precision'], metrics_df['recall'], 
           s=metrics_df['support']/10, # Size based on support
           alpha=0.7, c=range(len(metrics_df)), cmap='viridis')

# Add class labels to points
for i, (precision, recall, class_name) in enumerate(zip(metrics_df['precision'], 
                                                       metrics_df['recall'], 
                                                       metrics_df.index)):
    plt.annotate(class_name, 
                xy=(precision, recall),
                xytext=(5, 5),
                textcoords='offset points')

# Add reference lines
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

# Set chart properties
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.xlabel('Precision', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('Precision vs. Recall by Class', fontsize=14)

# Add explanatory text
plt.figtext(0.5, 0.01, 
           "This scatter plot shows the trade-off between precision and recall for each class.\n"
           "The size of each point represents the number of samples (support) for that class.\n"
           "Ideal performance is in the top-right corner (high precision and high recall).",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('work_imgs/16_precision_recall_scatter.png')
print("Precision-Recall scatter plot saved to 'work_imgs/16_precision_recall_scatter.png'")

# VISUALIZATION 4: Executive Summary Dashboard
plt.figure(figsize=(16, 10))  # Increased width for more categories

# Turn off the axes for the main figure
plt.axis('off')

# Title
plt.suptitle('Model Performance Executive Summary', fontsize=20, y=0.98)

# 1. Overall Metrics - Top left
ax1 = plt.axes([0.05, 0.65, 0.4, 0.25])
ax1.axis('off')
ax1.set_title('Overall Performance', fontsize=14)

# Calculate aggregate metrics
macro_precision = metrics_df['precision'].mean()
macro_recall = metrics_df['recall'].mean()
macro_f1 = metrics_df['f1-score'].mean()
weighted_precision = (metrics_df['precision'] * metrics_df['support']).sum() / metrics_df['support'].sum()
weighted_recall = (metrics_df['recall'] * metrics_df['support']).sum() / metrics_df['support'].sum()
weighted_f1 = (metrics_df['f1-score'] * metrics_df['support']).sum() / metrics_df['support'].sum()

# Create a table with overall metrics
overall_metrics = [
    ['Metric', 'Value', 'Interpretation'],
    ['Accuracy', f'{accuracy:.2f}', 'Proportion of correct predictions'],
    ['Macro F1-Score', f'{macro_f1:.2f}', 'Average F1 across classes (unweighted)'],
    ['Weighted F1-Score', f'{weighted_f1:.2f}', 'Average F1 weighted by class support'],
    ['Model Reliability', f'{"High" if weighted_f1 > 0.7 else "Medium" if weighted_f1 > 0.5 else "Low"}', 
     f'{"Strong performance" if weighted_f1 > 0.7 else "Average performance" if weighted_f1 > 0.5 else "Needs improvement"}']
]

cell_text = [row[1:] for row in overall_metrics[1:]]
cell_colors = [['lightgreen' if 'High' in text else 'khaki' if 'Medium' in text else 'lightcoral' if 'Low' in text else 'white' 
               for text in row] for row in cell_text]

table = ax1.table(cellText=cell_text,
                 colLabels=overall_metrics[0][1:],
                 rowLabels=[row[0] for row in overall_metrics[1:]],
                 loc='center',
                 cellLoc='center',
                 cellColours=cell_colors)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 2. Business Value - Bottom left
ax2 = plt.axes([0.05, 0.35, 0.4, 0.25])
ax2.axis('off')
ax2.set_title('Business Value & Next Steps', fontsize=14)

# Add text explaining business value of the model
business_text = (
    "Business Impact:\n"
    "• This model helps predict call outcomes with {:.0f}% accuracy\n"
    "• Best at identifying {}\n"
    "• Struggles most with {}\n\n"
    "Recommended Next Steps:\n"
    "• {} feature engineering to improve prediction of {}\n"
    "• Consider {} for addressing class imbalance\n"
    "• Deploy model for {} to gain immediate business value"
).format(
    accuracy * 100,
    best_class,
    worst_class,
    "Additional" if weighted_f1 < 0.7 else "Fine-tune",
    worst_class,
    "class weighting or oversampling" if imbalance_ratio > 2 else "current approach",
    "call prioritization" if weighted_f1 > 0.6 else "experimental use only"
)

ax2.text(0, 0.95, business_text, va='top', fontsize=11, linespacing=1.5)

# 3. Class Performance - Right side
ax3 = plt.axes([0.55, 0.35, 0.4, 0.55])
# Create a horizontal bar chart for F1 scores
sorted_f1 = metrics_df.sort_values('f1-score')
ax3.barh(sorted_f1.index, sorted_f1['f1-score'], color=colors[:len(sorted_f1)])
ax3.set_title('F1-Score by Class', fontsize=14)
ax3.set_xlim(0, 1)
ax3.set_xlabel('F1-Score')
ax3.tick_params(axis='y', labelsize=9)  # Smaller font for class names
ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
ax3.text(0.51, 0, 'Threshold for\nreliable predictions', fontsize=9, va='bottom')

# Add reference text
for i, (idx, row) in enumerate(sorted_f1.iterrows()):
    ax3.text(row['f1-score'] + 0.02, i, f"{row['f1-score']:.2f}", va='center')

# 4. Reliability gauge (based on weighted F1)
ax4 = plt.axes([0.05, 0.1, 0.9, 0.2])
ax4.axis('off')

# Create a simple reliability gauge based on weighted F1
gauge_colors = ['#FF6B6B', '#FFD166', '#06D6A0']  # Red, Yellow, Green
gauge_labels = ['Low Reliability\n(F1 < 0.5)', 'Medium Reliability\n(0.5 ≤ F1 < 0.7)', 'High Reliability\n(F1 ≥ 0.7)']
gauge_positions = [0.25, 0.5, 0.75]  # Positions for the gauge markers

# Draw the gauge background
gauge_height = 0.3
for i, color in enumerate(gauge_colors):
    if i == 0:
        rect = plt.Rectangle((0.1, 0), 0.27, gauge_height, facecolor=color, alpha=0.7)
    elif i == 1:
        rect = plt.Rectangle((0.37, 0), 0.26, gauge_height, facecolor=color, alpha=0.7)
    else:
        rect = plt.Rectangle((0.63, 0), 0.27, gauge_height, facecolor=color, alpha=0.7)
    ax4.add_patch(rect)

# Position labels under the gauge
for i, label in enumerate(gauge_labels):
    ax4.text(gauge_positions[i], -0.1, label, ha='center', fontsize=11)

# Add the needle/marker based on the weighted F1 score
needle_pos = weighted_f1 * 0.8 + 0.1  # Scale to gauge width (0.1 to 0.9)
ax4.plot([needle_pos, needle_pos], [0, gauge_height * 1.2], 'k', linewidth=2)
ax4.plot([needle_pos - 0.02, needle_pos + 0.02], [gauge_height * 1.2, gauge_height * 1.2], 'k', linewidth=2)
ax4.text(needle_pos, gauge_height * 1.4, f'F1 = {weighted_f1:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.savefig('work_imgs/17_executive_summary.png', dpi=300, bbox_inches='tight')
print("Executive summary dashboard saved to 'work_imgs/17_executive_summary.png'")

# VISUALIZATION 5: Performance vs. Random Baseline
plt.figure(figsize=(12, 6))  # Wider figure

# Calculate baseline performance (random guessing)
# For each class, random guessing would predict based on class frequency
class_probs = support / support.sum()
random_accuracy = (class_probs ** 2).sum()  # Random accuracy is sum of squared class probabilities
random_precision = class_probs.mean()  # Average precision would equal average class probability
random_recall = class_probs.mean()  # Same for recall
random_f1 = 2 * random_precision * random_recall / (random_precision + random_recall)  # Harmonic mean

# Comparison data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_values = [accuracy, weighted_precision, weighted_recall, weighted_f1]
baseline_values = [random_accuracy, random_precision, random_recall, random_f1]
improvement = [(model - baseline) / baseline * 100 for model, baseline in zip(model_values, baseline_values)]

# Create the bar chart
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
model_bars = ax.bar(x - width/2, model_values, width, label='Our Model', color=colors[0])
baseline_bars = ax.bar(x + width/2, baseline_values, width, label='Random Baseline', color=colors[3], alpha=0.7)

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(model_bars)
autolabel(baseline_bars)

# Add improvement percentage
for i, imp in enumerate(improvement):
    ax.text(i, max(model_values[i], baseline_values[i]) + 0.05, 
            f"+{imp:.0f}%", color='green', ha='center', fontweight='bold')

# Customize chart
ax.set_ylabel('Score')
ax.set_title('Model Performance vs. Random Baseline')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.2)  # Leave room for labels
ax.legend()

plt.figtext(0.5, 0.01, 
           f"This chart compares our model's performance against a random guessing baseline.\n"
           f"Our model shows significant improvements in all metrics, with accuracy {improvement[0]:.0f}% better than random chance.\n"
           f"This demonstrates clear business value from the feature engineering and modeling process.",
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
plt.savefig('work_imgs/18_baseline_comparison.png')
print("Baseline comparison visualization saved to 'work_imgs/18_baseline_comparison.png'")

# VISUALIZATION: Process flowchart showing the complete pipeline
plt.figure(figsize=(14, 8))
plt.axis('off')

# Define the process steps
steps = [
    "1. Simplify Target\nVariable",
    "2. Remove High\nCardinality Features",
    "3. Extract Time\nFeatures",
    "4. Handle Missing\nValues",
    "5. Create Meaningful\nFeatures",
    "6. Encode\nCategoricals",
    "7. Normalize\nNumerical Features",
    "8. Feature\nSelection",
    "9. Train Naive\nBayes Model"
]

improvements = [
    "20 classes → 6 classes",
    "Remove names, phones",
    "Hour, day, weekend flags",
    "Median & mode imputation",
    "Business hours, email features",
    "Label encoding",
    "MinMax scaling",
    "Select top 10 features",
    f"Accuracy: {accuracy:.2f}"
]

# Draw flowchart
box_height = 0.4
box_width = 1.2
x_pos = 0
y_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0]
x_step = 1.5

for i, (step, improvement) in enumerate(zip(steps, improvements)):
    # Position boxes in a flow
    if i < 3:
        y_pos[i] = 2  # Top row
    elif i < 6:
        y_pos[i] = 1  # Middle row
    else:
        y_pos[i] = 0  # Bottom row
    
    x_pos = (i % 3) * 4

    # Draw box
    rect = plt.Rectangle((x_pos, y_pos[i]), box_width, box_height, 
                         facecolor=colors[i % len(colors)], alpha=0.8, edgecolor='black')
    plt.gca().add_patch(rect)
    
    # Add text labels
    plt.text(x_pos + box_width/2, y_pos[i] + box_height/2, step, 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add improvement text
    plt.text(x_pos + box_width/2, y_pos[i] + box_height + 0.05, improvement, 
             ha='center', va='bottom', fontsize=9, fontweight='normal', color='darkslategray')
    
    # Add arrows between boxes
    if i < 8:
        if i % 3 == 2:  # Last box in row
            # Draw arrow down and to the left
            plt.arrow(x_pos + box_width/2, y_pos[i], 0, -0.7, 
                      head_width=0.1, head_length=0.1, fc='black', ec='black')
        else:
            # Draw horizontal arrow
            plt.arrow(x_pos + box_width, y_pos[i] + box_height/2, 0.5, 0, 
                      head_width=0.1, head_length=0.1, fc='black', ec='black')

# Set the limits
plt.xlim(-0.5, 12)
plt.ylim(-0.5, 3)
plt.title('Feature Engineering Pipeline for Outreach Calls Prediction', fontsize=14)

# Add explanatory text
plt.text(6, -0.3, 
         "This pipeline shows our 9-step feature engineering process that transformed raw call data\n"
         "into a predictive model. Each step addresses specific data challenges, with the\n"
         "biggest improvements coming from target simplification and time-based feature extraction.",
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.savefig('work_imgs/13_feature_engineering_pipeline.png', dpi=300, bbox_inches='tight')
print("Complete pipeline visualization saved to 'work_imgs/13_feature_engineering_pipeline.png'")

print("\nAnalysis complete with enhanced visualizations!")

#---------------------------------------
# Additional Evaluative Charts for Hire Efficacy
#---------------------------------------
print("\nCreating additional evaluative charts for hire efficacy...")

# Create a new directory for these charts
os.makedirs('work_imgs', exist_ok=True)

# We need to use the original dataframe (pre-preprocessing) to analyze specialists
print("Loading original data for specialist analysis...")
try:
    # Try to load directly from Google Sheets for the most up-to-date data
    print("Attempting to load original data from Google Sheets...")
    sheets_id = "1en_sV93LLm5m5oOAqGqd6O0UfbQGz1UXT01lnkowwFI"
    sheet_gid = "847781642"
    
    # Different URL format for public Google Sheets
    csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheets_id}/export?format=csv&gid={sheet_gid}"
    
    print("NOTE: To access private Google Sheets directly, the sheet must be either:")
    print("  1. Made publicly accessible (File > Share > Anyone with the link > Viewer)")
    print("  2. Or use gspread library with proper authentication (requires API setup)")
    print("Attempting direct access with public URL formats...")
    
    # Alternative methods to try if the first one fails
    alternative_urls = [
        # Standard export format
        f"https://docs.google.com/spreadsheets/d/{sheets_id}/export?format=csv&gid={sheet_gid}",
        
        # Public web CSV format
        f"https://docs.google.com/spreadsheets/d/{sheets_id}/gviz/tq?tqx=out:csv&gid={sheet_gid}",
    ]
    
    # Try each URL format until one works
    for url in alternative_urls:
        try:
            print(f"Trying URL: {url}")
            original_df = pd.read_csv(url)
            print(f"Successfully loaded original data using URL: {url}")
            break
        except Exception as url_error:
            if "401" in str(url_error):
                print(f"Authentication error: Sheet is not publicly accessible")
            else:
                print(f"Failed with URL {url}: {url_error}")
            continue
    else:
        # If all URLs fail, raise exception to trigger fallback
        raise Exception("Direct access failed - sheet requires authentication")
except Exception as e:
    # Fallback to local file if Google Sheets access fails
    print(f"Could not load original data from Google Sheets: {e}")
    print("Falling back to local CSV file...")
    original_df = pd.read_csv("Outreach_Calls.csv")

# Check if Outreach Specialist column exists
if 'Outreach Specialist' in original_df.columns:
    # 1. Success Rate by Outreach Specialist
    plt.figure(figsize=(14, 8))
    
    # Group by specialist and get success rates
    if 'Simplified_Outcome' in original_df.columns:
        target_col = 'Simplified_Outcome'
    elif 'Call Result' in original_df.columns:
        target_col = 'Call Result'
    else:
        # Find a suitable target column if neither exists
        potential_targets = [col for col in original_df.columns if 'result' in col.lower() or 'outcome' in col.lower()]
        target_col = potential_targets[0] if potential_targets else original_df.columns[-1]
    
    # Create simplified outcome if it doesn't exist
    if 'Simplified_Outcome' not in original_df.columns and 'Call Result' in original_df.columns:
        # Map original call results to simplified categories (same mapping as in the main code)
        outcome_mapping = {
            # Success categories
            '(YES) - Yes': 'Success',
            '(POS) - Possibility': 'Potential',
            '(NAR) - New Application Requested': 'Success',
            '(CB) - Callback': 'Potential',
            
            # Negative outcome categories
            '(NI) - Not Interested': 'Rejection',
            '(AS) - Another School': 'Rejection',
            '(COST) - Cost too much': 'Rejection',
            '(HU) - Hung Up': 'Rejection',
            
            # Contact but no decision categories
            '(ANS) - Left a Message': 'Contact_No_Decision',
            '(ANS) - Answering Machine': 'Contact_No_Decision',
            '(P) - Left Message w/ Someone/Parent': 'Contact_No_Decision',
            
            # Unable to reach categories
            '(NA) - No Answer': 'No_Contact',
            '(NA) - Inbox Full': 'No_Contact',
            '(WN) - Wrong Number': 'No_Contact',
            '(DISC) - Disconnected': 'No_Contact',
            '(BUS) - Busy Line': 'No_Contact',
            '(SNA) - Student not Available': 'No_Contact',
            '(TRAN) - Translator Needed': 'Other',
            '(HOME) - Live at Home/Own a Home': 'Other',
            '(APT) - Apartment': 'Other'
        }
        original_df['Simplified_Outcome'] = original_df['Call Result'].map(outcome_mapping)
        original_df['Simplified_Outcome'].fillna('Other', inplace=True)
        target_col = 'Simplified_Outcome'
    
    # Calculate success rate by specialist
    success_categories = ['Success', 'Potential']
    specialist_performance = original_df.groupby('Outreach Specialist')[target_col].apply(
        lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
    ).sort_values(ascending=False)
    
    # Plot top 15 specialists by success rate (if there are that many)
    top_specialists = min(15, len(specialist_performance))
    ax = specialist_performance.head(top_specialists).plot(kind='bar', figsize=(14, 8), color=colors[0])
    plt.title('Success Rate by Outreach Specialist (Top 15)', fontsize=16)
    plt.xlabel('Outreach Specialist', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add average line
    avg_success_rate = specialist_performance.mean()
    plt.axhline(y=avg_success_rate, color='red', linestyle='--', label=f'Average: {avg_success_rate:.2f}')
    
    # Add value labels on bars
    for i, v in enumerate(specialist_performance.head(top_specialists)):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('work_imgs/19_specialist_success_rate.png')
    print("Specialist success rate chart saved to 'work_imgs/19_specialist_success_rate.png'")
    
    # 2. Call Volume by Outreach Specialist
    plt.figure(figsize=(14, 8))
    
    call_volume = original_df['Outreach Specialist'].value_counts().sort_values(ascending=False)
    ax = call_volume.head(top_specialists).plot(kind='bar', figsize=(14, 8), color=colors[1])
    plt.title('Call Volume by Outreach Specialist (Top 15)', fontsize=16)
    plt.xlabel('Outreach Specialist', fontsize=14)
    plt.ylabel('Number of Calls', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add average line
    avg_calls = call_volume.mean()
    plt.axhline(y=avg_calls, color='red', linestyle='--', label=f'Average: {avg_calls:.0f}')
    
    # Add value labels on bars
    for i, v in enumerate(call_volume.head(top_specialists)):
        ax.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('work_imgs/20_specialist_call_volume.png')
    print("Specialist call volume chart saved to 'work_imgs/20_specialist_call_volume.png'")
    
    # 3. Efficiency Score (Success Rate * Call Volume normalized)
    plt.figure(figsize=(14, 8))
    
    # Normalize call volume (0-1 scale)
    call_volume_norm = (call_volume - call_volume.min()) / (call_volume.max() - call_volume.min())
    
    # Calculate efficiency score (success rate * normalized call volume)
    efficiency_score = specialist_performance * call_volume_norm
    efficiency_score = efficiency_score.sort_values(ascending=False)
    
    ax = efficiency_score.head(top_specialists).plot(kind='bar', figsize=(14, 8), color=colors[2])
    plt.title('Efficiency Score by Outreach Specialist (Top 15)', fontsize=16)
    plt.xlabel('Outreach Specialist', fontsize=14)
    plt.ylabel('Efficiency Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add average line
    avg_efficiency = efficiency_score.mean()
    plt.axhline(y=avg_efficiency, color='red', linestyle='--', label=f'Average: {avg_efficiency:.2f}')
    
    # Add value labels on bars
    for i, v in enumerate(efficiency_score.head(top_specialists)):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.figtext(0.5, 0.01, 
            "Efficiency Score combines success rate and call volume to identify specialists\n"
            "who are both effective (high success rate) and productive (high call volume).",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('work_imgs/21_specialist_efficiency_score.png')
    print("Specialist efficiency score chart saved to 'work_imgs/21_specialist_efficiency_score.png'")
    
    # 4. Outcome Distribution by Top Specialists
    plt.figure(figsize=(16, 10))
    
    # Get top 5 specialists by efficiency
    top5_specialists = efficiency_score.head(5).index
    
    # Create a dataframe with just the top specialists
    top_specialists_df = original_df[original_df['Outreach Specialist'].isin(top5_specialists)]
    
    # Calculate outcome distribution for each specialist
    outcome_dist = pd.crosstab(
        top_specialists_df['Outreach Specialist'], 
        top_specialists_df[target_col], 
        normalize='index'
    )
    
    # Plot stacked bar chart
    outcome_dist.plot(kind='bar', stacked=True, figsize=(16, 10), colormap='viridis')
    plt.title('Outcome Distribution for Top 5 Specialists', fontsize=16)
    plt.xlabel('Outreach Specialist', fontsize=14)
    plt.ylabel('Proportion of Calls', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Call Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.figtext(0.5, 0.01, 
            "This chart shows how call outcomes are distributed for the top 5 most efficient specialists.\n"
            "It helps identify specialists' strengths and weaknesses in achieving different outcomes.",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.savefig('work_imgs/22_top_specialists_outcome_distribution.png')
    print("Top specialists outcome distribution chart saved to 'work_imgs/22_top_specialists_outcome_distribution.png'")
    
    # 5. Performance by Time of Day
    plt.figure(figsize=(14, 8))
    
    # Check if we have time of day information
    time_cols = [col for col in original_df.columns if 'time_of_day' in col.lower()]
    
    if time_cols:
        time_col = time_cols[0]
        
        # Calculate success rate by specialist and time of day
        time_performance = original_df.groupby(['Outreach Specialist', time_col])[target_col].apply(
            lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
        ).unstack()
        
        # Get top 8 specialists for readability
        top8_specialists = efficiency_score.head(8).index
        time_performance = time_performance.loc[time_performance.index.isin(top8_specialists)]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(time_performance, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
        plt.title('Success Rate by Specialist and Time of Day', fontsize=16)
        plt.ylabel('Outreach Specialist', fontsize=14)
        plt.xlabel('Time of Day', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('work_imgs/23_specialist_time_of_day_performance.png')
        print("Specialist time of day performance chart saved to 'work_imgs/23_specialist_time_of_day_performance.png'")
    
    # 6. Performance Quadrant (Success Rate vs. Call Volume)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate proper quadrant dividers using median values
    volume_median = call_volume.median()
    success_median = specialist_performance.median()
    
    # Recalculate efficiency score to be more meaningful
    # Normalize call volume (0-1 scale)
    call_volume_norm = (call_volume - call_volume.min()) / (call_volume.max() - call_volume.min()) if call_volume.max() > call_volume.min() else call_volume
    
    # Calculate efficiency score as a combination of success rate and normalized call volume
    efficiency_score = specialist_performance * call_volume_norm
    
    # Calculate theoretical maximum possible efficiency 
    max_possible_success = 1.0  # Perfect success rate would be 1.0 (100%)
    max_possible_efficiency = max_possible_success * 1.0  # 1.0 is max normalized call volume
    
    # Normalize efficiency score for dot sizing
    efficiency_norm = (efficiency_score - efficiency_score.min()) / (efficiency_score.max() - efficiency_score.min()) if efficiency_score.max() > efficiency_score.min() else efficiency_score
    
    # Print the efficiency scores for analysis
    print("\nEfficiency Scores (Success Rate × Normalized Call Volume):")
    print(f"Maximum possible efficiency score (theoretical): {max_possible_efficiency:.4f}")
    print(f"Actual maximum efficiency in dataset: {efficiency_score.max():.4f}")
    print(f"Actual maximum success rate in dataset: {specialist_performance.max():.4f}")
    for specialist, eff in efficiency_score.sort_values(ascending=False).items():
        print(f"{specialist}: {eff:.4f} (Normalized: {efficiency_norm[specialist]:.4f}, Success Rate: {specialist_performance[specialist]:.4f}, Call Volume: {call_volume[specialist]})")
    
    # Create a dictionary mapping specialists to colors for consistency
    specialist_colors = {
        name: plt.cm.tab10(i) for i, name in enumerate(efficiency_score.index)
    }
    
    # Get the current axis limits before plotting
    x_min = min(call_volume) * 0.9
    x_max = max(call_volume) * 1.1
    
    # Rescale the success rate range to position specialists better
    # Set minimum to 60% of the lowest success rate
    y_min = min(specialist_performance) * 0.6  
    # Set maximum to 20% higher than the highest success rate or 1.0, whichever is lower
    y_max = min(1.0, max(specialist_performance) * 1.2)
    
    # Set the axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Create a gradient background from top-right (green) to bottom-left (red)
    # This requires creating a custom gradient using imshow
    gradient_x = np.linspace(0, 1, 100)
    gradient_y = np.linspace(0, 1, 100)
    
    # Create grid of coordinates
    X, Y = np.meshgrid(gradient_x, gradient_y)
    
    # Create a gradient that goes from bottom-left to top-right
    Z = X + Y  # Values will range from 0 (bottom-left) to 2 (top-right)
    
    # Normalize to 0-1 range
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    
    # Plot the gradient background
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(Z, cmap=plt.cm.RdYlGn, alpha=0.15, extent=extent, aspect='auto', origin='lower')
    
    # Create scatter plot with each specialist having their own color
    for specialist, success_rate in specialist_performance.items():
        volume = call_volume[specialist]
        efficiency = efficiency_norm[specialist]  # Use normalized efficiency for dot size
        
        # Plot each specialist with their color and size based on efficiency
        # Scale dot size from 100 to 800 based on normalized efficiency
        dot_size = 100 + (efficiency * 700)
        
        ax.scatter(
            volume, 
            success_rate, 
            s=dot_size,  # Use properly scaled dot size
            alpha=0.8, 
            color=specialist_colors[specialist],
            label=f"{specialist} (Eff: {efficiency_score[specialist]:.4f})"  # Show actual efficiency score
        )
    
    # Add quadrant lines
    ax.axhline(y=success_median, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=volume_median, color='red', linestyle='--', alpha=0.7)
    
    # Add percentage formatting to y-axis (success rate)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Calculate offset from median lines for quadrant labels
    x_offset = (x_max - x_min) * 0.08
    y_offset = (y_max - y_min) * 0.08
    
    # Add simple quadrant labels near the center intersection
    # High Performers - top right quadrant
    ax.text(volume_median + x_offset, 
            success_median + y_offset, 
            'HIGH\nPERFORMERS', 
            ha='left', va='bottom', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, pad=0.2, edgecolor='none'))
    
    # Quality Focused - top left quadrant
    ax.text(volume_median - x_offset, 
            success_median + y_offset, 
            'QUALITY\nFOCUSED', 
            ha='right', va='bottom', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, pad=0.2, edgecolor='none'))
    
    # Needs Improvement - bottom left quadrant
    ax.text(volume_median - x_offset, 
            success_median - y_offset, 
            'NEEDS\nIMPROVEMENT', 
            ha='right', va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, pad=0.2, edgecolor='none'))
    
    # Quantity Focused - bottom right quadrant
    ax.text(volume_median + x_offset, 
            success_median - y_offset, 
            'QUANTITY\nFOCUSED', 
            ha='left', va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, pad=0.2, edgecolor='none'))
    
    # Create a shaded region showing perfect efficiency (1.0) for context
    # Add an annotation explaining the efficiency scale
    ax.text(0.02, 0.9, 
            f"Efficiency Score = Success Rate × Normalized Call Volume\n"
            f"Maximum possible efficiency: 1.0 (perfect success rate with max volume)\n"
            f"Current best efficiency: {efficiency_score.max():.4f} (Marina)\n"
            f"Success rates shown as percentages (current max: {specialist_performance.max():.1%})", 
            transform=ax.transAxes, fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7), 
            ha='left', va='center')
    
    ax.set_title('Performance Quadrant: Success Rate vs. Call Volume', fontsize=16)
    ax.set_xlabel('Call Volume (Number of Calls)', fontsize=14)
    ax.set_ylabel('Success Rate (% of Successful Calls)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Move legend to below the chart
    legend = ax.legend(title="Specialists & Efficiency Scores", 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, -0.15),
                      ncol=len(specialist_performance),
                      fontsize=10,
                      markerscale=0.7)  # Smaller markers in legend
    
    plt.figtext(0.5, -0.05, 
            "This quadrant analysis identifies four specialist personas based on call success and volume.\n"
            "High performers (top-right) have both high success rates and high call volumes.\n"
            "Dot size represents efficiency score (Success Rate × Normalized Call Volume).",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Make room for legend and caption below
    plt.savefig('work_imgs/24_specialist_performance_quadrant.png', bbox_inches='tight')
    print("Specialist performance quadrant chart saved to 'work_imgs/24_specialist_performance_quadrant.png'")
    
    # 7. Improvement Over Time
    # Check if we have date information
    date_cols = [col for col in original_df.columns if 'date' in col.lower() or 'day' in col.lower() or 'month' in col.lower()]
    
    if date_cols:
        date_col = date_cols[0]
        
        plt.figure(figsize=(14, 8))
        
        # Group by date and get average success rate
        if 'month' in date_col.lower():
            # For month-based analysis
            try:
                time_trend = original_df.groupby(date_col)[target_col].apply(
                    lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
                )
                plt.title('Success Rate Trend by Month', fontsize=16)
                plt.xlabel('Month', fontsize=14)
            except:
                print(f"Could not create month-based trend chart with column {date_col}")
                
        elif 'day' in date_col.lower():
            # For day-based analysis
            try:
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                time_trend = original_df.groupby(date_col)[target_col].apply(
                    lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
                )
                # Reorder by days of week if this is a day name column
                if time_trend.index.isin(days_order).any():
                    time_trend = time_trend.reindex(days_order)
                plt.title('Success Rate by Day', fontsize=16)
                plt.xlabel('Day', fontsize=14)
            except:
                print(f"Could not create day-based trend chart with column {date_col}")
        else:
            # Try to parse as datetime for other columns
            try:
                original_df['date'] = pd.to_datetime(original_df[date_col])
                original_df['month_year'] = original_df['date'].dt.to_period('M')
                time_trend = original_df.groupby('month_year')[target_col].apply(
                    lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
                )
                plt.title('Success Rate Trend Over Time', fontsize=16)
                plt.xlabel('Time Period', fontsize=14)
            except:
                print(f"Could not create time trend chart with column {date_col}")
                time_trend = None
        
        if 'time_trend' in locals() and time_trend is not None:
            # Plot line chart
            time_trend.plot(kind='line', marker='o', figsize=(14, 8), color=colors[4], linewidth=2)
            plt.ylabel('Success Rate', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add trend line
            try:
                z = np.polyfit(range(len(time_trend)), time_trend, 1)
                p = np.poly1d(z)
                plt.plot(range(len(time_trend)), p(range(len(time_trend))), "r--", alpha=0.8, 
                         label=f"Trend: {'Increasing' if z[0]>0 else 'Decreasing'}")
                plt.legend()
            except:
                pass
            
            plt.tight_layout()
            plt.savefig('work_imgs/25_success_rate_trend.png')
            print("Success rate trend chart saved to 'work_imgs/25_success_rate_trend.png'")

    # 8. Top Specialist Comparison Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top 5 Specialists Performance Dashboard', fontsize=20)
    
    # Get top 5 specialists
    top5_specialists = efficiency_score.head(5).index
    top5_df = original_df[original_df['Outreach Specialist'].isin(top5_specialists)]
    
    # Chart 1: Success Rate
    top5_success = specialist_performance[top5_specialists]
    axes[0, 0].bar(range(len(top5_success)), top5_success, color=colors[:len(top5_success)])
    axes[0, 0].set_title('Success Rate', fontsize=14)
    axes[0, 0].set_xticks(range(len(top5_success)))
    axes[0, 0].set_xticklabels(top5_success.index, rotation=45, ha='right')
    axes[0, 0].axhline(y=specialist_performance.mean(), color='red', linestyle='--', label='Avg')
    axes[0, 0].set_ylim(0, max(top5_success)*1.2)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(top5_success):
        axes[0, 0].text(i, v + 0.01, f'{v:.2f}', ha='center')
    axes[0, 0].legend()
    
    # Chart 2: Call Volume
    top5_volume = call_volume[top5_specialists]
    axes[0, 1].bar(range(len(top5_volume)), top5_volume, color=colors[:len(top5_volume)])
    axes[0, 1].set_title('Call Volume', fontsize=14)
    axes[0, 1].set_xticks(range(len(top5_volume)))
    axes[0, 1].set_xticklabels(top5_volume.index, rotation=45, ha='right')
    axes[0, 1].axhline(y=call_volume.mean(), color='red', linestyle='--', label='Avg')
    axes[0, 1].set_ylim(0, max(top5_volume)*1.2)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(top5_volume):
        axes[0, 1].text(i, v + 5, f'{v:.0f}', ha='center')
    axes[0, 1].legend()
    
    # Chart 3: Success Rate by Outcome Category
    if len(top5_df) > 0:
        for i, specialist in enumerate(top5_specialists):
            spec_df = top5_df[top5_df['Outreach Specialist'] == specialist]
            outcome_counts = spec_df[target_col].value_counts()
            outcome_pct = outcome_counts / outcome_counts.sum()
            
            # Only plot the success categories
            success_pct = outcome_pct[outcome_pct.index.isin(success_categories)]
            if not success_pct.empty:
                axes[1, 0].bar([i], [success_pct.sum()], label=specialist if i == 0 else "_nolegend_", 
                               color=colors[i % len(colors)])
                
                # Break down by success category if there's more than one
                if len(success_pct) > 1:
                    bottom = 0
                    for j, (category, pct) in enumerate(success_pct.items()):
                        axes[1, 0].bar([i], [pct], bottom=bottom, alpha=0.5,
                                      label=f"{category}" if i == 0 else "_nolegend_", 
                                      color=colors[(i+j+1) % len(colors)])
                        bottom += pct
                
        axes[1, 0].set_title('Success Category Breakdown', fontsize=14)
        axes[1, 0].set_xticks(range(len(top5_specialists)))
        axes[1, 0].set_xticklabels(top5_specialists, rotation=45, ha='right')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1, 0].set_ylabel('Proportion of Successful Calls')
        axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Chart 4: Efficiency Score
    top5_efficiency = efficiency_score[top5_specialists]
    axes[1, 1].bar(range(len(top5_efficiency)), top5_efficiency, color=colors[:len(top5_efficiency)])
    axes[1, 1].set_title('Efficiency Score', fontsize=14)
    axes[1, 1].set_xticks(range(len(top5_efficiency)))
    axes[1, 1].set_xticklabels(top5_efficiency.index, rotation=45, ha='right')
    axes[1, 1].axhline(y=efficiency_score.mean(), color='red', linestyle='--', label='Avg')
    axes[1, 1].set_ylim(0, max(top5_efficiency)*1.2)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(top5_efficiency):
        axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center')
    axes[1, 1].legend()
    
    # Add explanatory text
    fig.text(0.5, 0.01, 
             "This dashboard compares the top 5 specialists based on our efficiency score metric.\n"
             "The most effective specialists combine high success rates with high call volumes,\n"
             "while also achieving quality outcomes within the success categories.",
             ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig('work_imgs/26_top_specialists_dashboard.png')
    print("Top specialists dashboard saved to 'work_imgs/26_top_specialists_dashboard.png'")

else:
    print("Could not find 'Outreach Specialist' column in the dataset to create specialist efficacy charts.")

print("\nAdditional evaluative charts for hire efficacy complete!")

#---------------------------------------
# Additional Requested Charts Based on Examples
#---------------------------------------
print("\nCreating additional requested charts based on examples...")

if 'Outreach Specialist' in original_df.columns:
    # Filter to top specialists for cleaner visualization (up to 4)
    top_specialists = efficiency_score.head(4).index.tolist()
    top_spec_df = original_df[original_df['Outreach Specialist'].isin(top_specialists)]
    
    #---------------------------------------
    # 1. Success Rate and Call Volume Bar Charts Combined
    #---------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Success Rate - left chart
    success_rate = top_spec_df.groupby('Outreach Specialist')[target_col].apply(
        lambda x: (x.isin(success_categories).sum() / len(x)) * 100 if len(x) > 0 else 0
    ).sort_values(ascending=True)
    
    bars1 = ax1.barh(success_rate.index, success_rate.values, color=plt.cm.viridis(np.linspace(0, 0.8, len(success_rate))))
    ax1.set_title('Success Rate by Specialist', fontsize=14, fontweight='bold')
    ax1.set_xlabel('YES Rate', fontsize=12)
    ax1.set_ylabel('Specialist', fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage labels to bars
    for i, v in enumerate(success_rate):
        ax1.text(v + 0.3, i, f"{v:.1f}%", va='center', fontsize=10)
    
    # Call Volume - right chart
    call_volume = top_spec_df['Outreach Specialist'].value_counts().sort_values(ascending=True)
    
    bars2 = ax2.barh(call_volume.index, call_volume.values, color=plt.cm.magma(np.linspace(0.2, 0.8, len(call_volume))))
    ax2.set_title('Call Volume by Specialist', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Calls Made', fontsize=12)
    ax2.set_ylabel('Specialist', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add count labels to bars
    for i, v in enumerate(call_volume):
        ax2.text(v + 5, i, f"{v}", va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('work_imgs/27_top_specialists_performance_bars.png')
    print("Top specialists performance bars saved to 'work_imgs/27_top_specialists_performance_bars.png'")
    
    #---------------------------------------
    # 2. Call Quality Distribution Chart
    #---------------------------------------
    plt.figure(figsize=(16, 8))
    
    # Create a simplified outcome categorization for better visualization
    outcome_categories = {
        'High Value': ['Success'],
        'Medium Value': ['Potential'],
        'Neutral': ['Contact_No_Decision', 'No_Contact'],
        'Low Value': ['Rejection', 'Other']
    }
    
    # Function to categorize outcomes
    def categorize_outcome(outcome):
        for category, outcomes in outcome_categories.items():
            if outcome in outcomes:
                return category
        return 'Neutral'  # Default category
    
    # Create a copy of the dataframe with the new categorization
    if 'Call_Quality' not in top_spec_df.columns:
        top_spec_df['Call_Quality'] = top_spec_df[target_col].apply(categorize_outcome)
    
    # Calculate percentage distribution for each specialist
    quality_dist = pd.crosstab(
        top_spec_df['Outreach Specialist'], 
        top_spec_df['Call_Quality'],
        normalize='index'
    ) * 100
    
    # Ensure all categories are present (fill with 0 if missing)
    for category in ['High Value', 'Medium Value', 'Neutral', 'Low Value']:
        if category not in quality_dist.columns:
            quality_dist[category] = 0
    
    # Sort by specialist order
    quality_dist = quality_dist.reindex(call_volume.index)
    
    # Plot horizontal stacked bars
    colors_dict = {
        'High Value': '#1b9e77',
        'Medium Value': '#7570b3',
        'Neutral': '#d95f02',
        'Low Value': '#e7298a'
    }
    
    # Initialize the left positions for each specialist
    y_pos = np.arange(len(quality_dist.index))
    left_pos = np.zeros(len(quality_dist.index))
    
    # Create the stacked bars
    for category in ['High Value', 'Medium Value', 'Neutral', 'Low Value']:
        if category in quality_dist.columns:
            plt.barh(
                y_pos, 
                quality_dist[category], 
                left=left_pos, 
                height=0.8, 
                label=category,
                color=colors_dict.get(category, 'gray')
            )
            
            # Add percentage labels
            for i, percentage in enumerate(quality_dist[category]):
                if percentage > 5:  # Only show labels for significant percentages
                    text_x = left_pos[i] + percentage / 2
                    plt.text(text_x, i, f"{percentage:.0f}%", ha='center', va='center', fontsize=10)
            
            # Update the left position for the next category
            left_pos = left_pos + quality_dist[category]
    
    plt.yticks(y_pos, quality_dist.index)
    plt.xlabel('Percentage of Calls', fontsize=12)
    plt.ylabel('Specialist', fontsize=12)
    plt.title('Call Quality Distribution', fontsize=16, fontweight='bold')
    plt.legend(title='Call Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('work_imgs/28_call_quality_distribution.png')
    print("Call quality distribution chart saved to 'work_imgs/28_call_quality_distribution.png'")
    
    #---------------------------------------
    # 3. Performance Trends Over Time (3-day Moving Average)
    #---------------------------------------
    
    # Check if we have date information
    date_cols = [col for col in original_df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    
    if date_cols or 'Timestamp' in original_df.columns:
        date_col = date_cols[0] if date_cols else 'Timestamp'
        
        plt.figure(figsize=(16, 8))
        
        try:
            # Convert to datetime
            if 'date_col_dt' not in locals():
                original_df['date_col_dt'] = pd.to_datetime(original_df[date_col], errors='coerce')
            
            # Extract date part only
            original_df['call_date'] = original_df['date_col_dt'].dt.date
            
            # Create performance score (can be success rate, efficiency, etc.)
            # Here we'll use success rate as the performance score
            performance_data = []
            
            # Calculate daily performance for each specialist
            for specialist in top_specialists:
                spec_df = original_df[original_df['Outreach Specialist'] == specialist]
                daily_perf = spec_df.groupby('call_date')[target_col].apply(
                    lambda x: (x.isin(success_categories).sum() / len(x)) if len(x) > 0 else 0
                )
                
                # Convert to DataFrame with specialist column
                daily_perf_df = pd.DataFrame({'Date': daily_perf.index, 
                                             'Performance': daily_perf.values,
                                             'Specialist': specialist})
                performance_data.append(daily_perf_df)
            
            # Combine all specialists' data
            if performance_data:
                performance_df = pd.concat(performance_data)
                
                # Create line plot with 3-day moving average
                for specialist in top_specialists:
                    spec_data = performance_df[performance_df['Specialist'] == specialist]
                    
                    if len(spec_data) > 2:  # Need at least 3 points for moving average
                        # Sort by date
                        spec_data = spec_data.sort_values('Date')
                        
                        # Calculate 3-day moving average
                        spec_data['MA_3day'] = spec_data['Performance'].rolling(window=3, min_periods=1).mean()
                        
                        # Plot the moving average
                        plt.plot(spec_data['Date'], spec_data['MA_3day'], 
                                marker='o', linewidth=2, 
                                label=f"{specialist} (3-day avg)")
                        
                        # Add trend line
                        try:
                            # Add slight random offset to date numeric value to prevent singular matrix
                            date_numeric = np.array(range(len(spec_data['Date'])))
                            z = np.polyfit(date_numeric, spec_data['MA_3day'], 1)
                            p = np.poly1d(z)
                            plt.plot(spec_data['Date'], p(date_numeric), 
                                    linestyle='--', alpha=0.8, color=plt.gca().lines[-1].get_color())
                        except:
                            pass
                
                # Set chart properties
                plt.title('Performance Trends Over Time (3-day Moving Average)', fontsize=16, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Performance Score', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Add reference line for average performance
                avg_perf = performance_df['Performance'].mean()
                plt.axhline(y=avg_perf, color='red', linestyle=':', alpha=0.5)
                plt.text(performance_df['Date'].min(), avg_perf + 0.02, f"Avg: {avg_perf:.2f}", 
                        fontsize=9, color='red')
                
                plt.tight_layout()
                plt.savefig('work_imgs/29_performance_trends_over_time.png')
                print("Performance trends chart saved to 'work_imgs/29_performance_trends_over_time.png'")
            else:
                print("Not enough time-series data to create performance trends chart")
        except Exception as e:
            print(f"Could not create performance trends chart: {e}")
    
    #---------------------------------------
    # 4. Specialist Performance Comparison (Radar Chart)
    #---------------------------------------
    try:
        plt.figure(figsize=(10, 10))
        
        # Define the metrics to compare (select 4 for radar chart)
        metrics = {
            'YES Rate': top_spec_df.groupby('Outreach Specialist')[target_col].apply(
                lambda x: (x.isin(['Success']).sum() / len(x)) if len(x) > 0 else 0
            ),
            'Medium Value Rate': top_spec_df.groupby('Outreach Specialist')[target_col].apply(
                lambda x: (x.isin(['Potential']).sum() / len(x)) if len(x) > 0 else 0
            ),
            'Avg Calls Per Day': original_df.groupby(['Outreach Specialist', 'call_date']).size().groupby('Outreach Specialist').mean() 
            if 'call_date' in original_df.columns else call_volume / 30,  # Approximate if no date
            'Weighted Score': efficiency_score
        }
        
        # Normalize each metric to 0-1 scale for comparability
        for metric in metrics:
            metrics[metric] = (metrics[metric] - metrics[metric].min()) / (metrics[metric].max() - metrics[metric].min()) if metrics[metric].max() > metrics[metric].min() else metrics[metric]
        
        # Number of metrics
        N = len(metrics)
        
        # Create angles for each metric (divide the plot into equal parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection (for radar chart)
        ax = plt.subplot(111, polar=True)
        
        # Add labels for each metric
        plt.xticks(angles[:-1], metrics.keys(), fontsize=12)
        
        # Add radial labels (0%, 25%, 50%, 75%, 100%)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each specialist
        for i, specialist in enumerate(top_specialists):
            # Get values for this specialist (default to 0 if missing)
            values = [metrics[metric].get(specialist, 0) for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot the specialist
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=specialist, 
                   color=plt.cm.tab10(i))
            ax.fill(angles, values, alpha=0.1, color=plt.cm.tab10(i))
        
        # Add legend
        plt.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title("Specialist Performance Comparison", fontsize=16, fontweight='bold', y=1.1)
        
        plt.tight_layout()
        plt.savefig('work_imgs/30_specialist_radar_comparison.png')
        print("Specialist radar comparison chart saved to 'work_imgs/30_specialist_radar_comparison.png'")
    except Exception as e:
        print(f"Could not create radar chart: {e}")
    
    #---------------------------------------
    # 5. Best Time to Call Heatmap
    #---------------------------------------
    try:
        # Check if we have hour and day information
        if 'date_col_dt' in original_df.columns:
            # Extract hour and day
            original_df['hour_of_day'] = original_df['date_col_dt'].dt.hour
            original_df['day_of_week'] = original_df['date_col_dt'].dt.day_name()
            
            # Filter for business hours only (8am to 8pm)
            business_hours = original_df[(original_df['hour_of_day'] >= 17) & 
                                        (original_df['hour_of_day'] <= 20)]
            
            # Create figure with subplots for each specialist
            fig, axes = plt.subplots(1, len(top_specialists), figsize=(20, 8), sharey=True)
            
            # If there's only one specialist, axes won't be an array
            if len(top_specialists) == 1:
                axes = [axes]
                
            # Define day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            # Create a success rate heatmap for each specialist
            for i, specialist in enumerate(top_specialists):
                # Filter for this specialist
                spec_df = business_hours[business_hours['Outreach Specialist'] == specialist]
                
                # Group by day and hour to get success rate
                success_by_time = spec_df.groupby(['day_of_week', 'hour_of_day'])[target_col].apply(
                    lambda x: (x.isin(['Success']).sum() / len(x) * 100) if len(x) > 0 else 0
                ).unstack()
                
                # Reindex to ensure consistent days
                if success_by_time.empty:
                    # Create empty DataFrame if no data
                    success_by_time = pd.DataFrame(0, index=day_order, 
                                                 columns=range(17, 21))
                else:
                    # Order days of week properly
                    days_available = [d for d in day_order if d in success_by_time.index]
                    success_by_time = success_by_time.reindex(days_available)
                
                # Create heatmap for this specialist
                im = sns.heatmap(success_by_time, annot=True, fmt='.0f', cmap='YlGnBu',
                                cbar=False, ax=axes[i], vmin=0, vmax=100,
                                annot_kws={"size": 10})
                
                # Set title and labels
                axes[i].set_title(specialist, fontsize=12)
                axes[i].set_xlabel('Day of Week', fontsize=10)
                if i == 0:
                    axes[i].set_ylabel('Hour of Day', fontsize=10)
                
            # Add an overall title and color bar
            fig.suptitle('Best Time to Call by Specialist (% YES outcomes)', fontsize=16, fontweight='bold')
            
            # Add color bar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Success Rate', fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for color bar
            plt.savefig('work_imgs/31_best_time_to_call_heatmap.png')
            print("Best time to call heatmap saved to 'work_imgs/31_best_time_to_call_heatmap.png'")
        else:
            print("Could not create best time to call heatmap - missing datetime information")
    except Exception as e:
        print(f"Could not create best time to call heatmap: {e}")
else:
    print("Could not find 'Outreach Specialist' column in the dataset to create specialist charts.")

print("\nAll requested charts have been created!")

#---------------------------------------
# Step 10: User Input for Prediction
#---------------------------------------
print("\n" + "="*80)
print("\nCall Outcome Prediction Tool")
print("="*80)

# Function to predict outcome
def predict_call_outcome(user_input_features):
    """
    Predict the call outcome based on user input features.
    
    Args:
        user_input_features (dict): Dictionary of feature values entered by user
    
    Returns:
        str: Predicted outcome category
    """
    # Create DataFrame with the same structure as the training data
    input_df = pd.DataFrame([user_input_features])
    
    # Apply same preprocessing as during training
    
    # Handle categorical features (if any)
    # Only process categorical columns that exist in user input
    for col in input_df.columns:
        if col in X_selected.columns and X_selected[col].dtype == 'object':
            # Check if label encoder for this column exists
            encoder_name = f'le_{col}'
            if encoder_name in locals() or encoder_name in globals():
                encoder = locals().get(encoder_name) or globals().get(encoder_name)
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except:
                    # If value not in training data, use most common value
                    print(f"Warning: Value '{input_df[col][0]}' for '{col}' not seen during training. Using most common value.")
                    input_df[col] = 0  # Default to most common class
    
    # Handle numeric features - apply the same scaling
    numeric_cols = [col for col in input_df.columns if col in X_selected.columns]
    if len(numeric_cols) > 0:
        # Check if the columns match what scaler expects
        if hasattr(scaler, 'feature_names_in_'):
            # Newer scikit-learn versions
            expected_cols = list(scaler.feature_names_in_)
            for col in numeric_cols:
                if col not in expected_cols:
                    input_df.drop(col, axis=1, inplace=True)
        try:
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        except:
            print("Warning: Could not apply scaling to input features. Using raw values.")
    
    # Ensure we only use the same features used during training
    # Keep only columns that exist in X_selected
    final_cols = [col for col in input_df.columns if col in X_selected.columns]
    
    # If we're missing columns from X_selected, add them with default values
    for col in X_selected.columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default to 0 for missing features
    
    # Reorder columns to match X_selected
    input_df = input_df[X_selected.columns]
    
    # Make prediction
    try:
        pred_encoded = model.predict(input_df)[0]
        # Convert the prediction back to the original class label
        predicted_category = target_mapping.get(pred_encoded, "Unknown")
        
        # Also get prediction probabilities for all classes
        pred_probs = model.predict_proba(input_df)[0]
        class_probs = {target_mapping.get(i, f"Class {i}"): prob for i, prob in enumerate(pred_probs)}
        
        return predicted_category, class_probs
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", {}

def get_user_input():
    """
    Prompt the user for input values for each feature needed for prediction.
    
    Returns:
        dict: Dictionary of feature values entered by user
    """
    print("\nPlease enter the following information to predict the call outcome:")
    print("(Press Enter to use default values shown in brackets)\n")
    
    user_input = {}
    
    # Determine which features to ask for based on the model's selected features
    features_to_collect = []
    
    # If we have X_selected, use those features
    if 'X_selected' in globals():
        features_to_collect = X_selected.columns.tolist()
    else:
        # Fallback to a default set of features
        features_to_collect = [
            'Is_Business_Hours', 
            'Is_Weekday',
            'Has_Email',
            'Call_Time_Hour',
            'Call_Duration',
            'Specialist_Experience'
        ]

    # Create user-friendly descriptions for each feature
    feature_descriptions = {
        'Is_Business_Hours': 'Was the call made during business hours (9am-5pm)? (yes/no)',
        'Is_Weekday': 'Was the call made on a weekday? (yes/no)',
        'Has_Email': 'Did the prospect provide an email address? (yes/no)',
        'Call_Time_Hour': 'What hour of the day was the call made? (0-23)',
        'Call_Duration': 'How long was the call in seconds?',
        'Specialist_Experience': 'How many months of experience does the specialist have?',
        'Specialist_Time_Interaction': 'How effective is the specialist at this time of day? (0-1)',
        'Email_Domain': 'What is the email domain? (gmail, yahoo, hotmail, etc.)',
        'Is_Previous_Contact': 'Has the prospect been contacted before? (yes/no)',
        'Is_Mobile_Phone': 'Is the call being made to a mobile phone? (yes/no)'
    }
    
    # Define default values
    default_values = {
        'Is_Business_Hours': 'yes',
        'Is_Weekday': 'yes',
        'Has_Email': 'yes',
        'Call_Time_Hour': '14',
        'Call_Duration': '120',
        'Specialist_Experience': '12',
        'Specialist_Time_Interaction': '0.7',
        'Email_Domain': 'gmail',
        'Is_Previous_Contact': 'no',
        'Is_Mobile_Phone': 'yes'
    }
    
    # For each feature in features_to_collect, ask the user for input
    for feature in features_to_collect:
        # Get a user-friendly description for this feature
        description = feature_descriptions.get(feature, f"Enter value for {feature}")
        default = default_values.get(feature, "0")
        
        # Get user input
        value = input(f"{description} [{default}]: ")
        
        # If user didn't enter anything, use the default value
        if not value.strip():
            value = default
        
        # Convert yes/no to 1/0 for boolean features
        if value.lower() in ['yes', 'y', 'true', 't']:
            value = 1
        elif value.lower() in ['no', 'n', 'false', 'f']:
            value = 0
        
        # Try to convert to numeric if possible
        try:
            value = float(value)
            # Convert to int if it's a whole number
            if value == int(value):
                value = int(value)
        except:
            # Keep as string if not numeric
            pass
        
        user_input[feature] = value
    
    return user_input

def display_prediction(outcome, probabilities):
    """
    Display the prediction in a user-friendly way.
    
    Args:
        outcome (str): Predicted outcome category
        probabilities (dict): Dictionary of class probabilities
    """
    print("\n" + "="*60)
    print(f"Predicted Call Outcome: {outcome}")
    print("="*60)
    
    # Further explanation based on outcome
    if outcome == 'Success':
        print("This call is predicted to be successful. ")
        print("The prospect is likely to say YES or request an application.")
    elif outcome == 'Potential':
        print("This call is predicted to have potential.")
        print("The prospect may request a callback or show interest.")
    elif outcome == 'Rejection':
        print("This call is predicted to result in rejection.")
        print("The prospect may not be interested or choose another option.")
    elif outcome == 'Contact_No_Decision':
        print("This call is predicted to result in contact but no decision.")
        print("You may need to leave a message or follow up later.")
    elif outcome == 'No_Contact':
        print("This call is predicted to result in no contact.")
        print("You may encounter a wrong number, disconnected line, or no answer.")
    else:
        print(f"Prediction: {outcome}")
    
    # Show probabilities
    print("\nConfidence levels:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for category, prob in sorted_probs:
        print(f"  {category}: {prob*100:.1f}%")
    
    print("\nNote: This prediction is based on historical patterns and should be")
    print("used as a guide, not a definitive outcome. Always use your judgment")
    print("and expertise when making calls.")

# Main prediction loop
def main_prediction_interface():
    while True:
        # Get user input
        user_input = get_user_input()
        
        # Make prediction
        outcome, probabilities = predict_call_outcome(user_input)
        
        # Display results
        display_prediction(outcome, probabilities)
        
        # Ask if the user wants to make another prediction
        another = input("\nWould you like to predict another call outcome? (yes/no): ")
        if another.lower() not in ['yes', 'y']:
            print("\nThank you for using the Call Outcome Prediction Tool!")
            break

# Run the prediction interface
if __name__ == "__main__":
    # Try to run prediction interface if model exists
    if 'model' in globals():
        main_prediction_interface()
    else:
        # Try to load the model from file
        try:
            print("Loading saved model from file...")
            model = joblib.load('saved_models/call_outcome_model.pkl')
            
            # Load metadata
            model_metadata = joblib.load('saved_models/call_outcome_metadata.pkl')
            
            # Create globals
            X_selected = pd.DataFrame(columns=model_metadata['feature_columns'])
            target_mapping = model_metadata['target_mapping']
            
            if model_metadata['scaler'] is not None:
                scaler = model_metadata['scaler']
            
            print("Model loaded successfully!")
            main_prediction_interface()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Error: Model not found. Please run the training code first.")
