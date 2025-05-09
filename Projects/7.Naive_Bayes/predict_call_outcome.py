import os
import pandas as pd
import numpy as np
import joblib

# ASCII art banner for a nicer UI
BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                 CALL OUTCOME PREDICTION SYSTEM                        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def predict_call_outcome(user_input_features, model, X_selected, target_mapping, scaler=None):
    """
    Predict the call outcome based on user input features.
    
    Args:
        user_input_features (dict): Dictionary of feature values entered by user
        model: The trained machine learning model
        X_selected: DataFrame with the features used by the model
        target_mapping: Mapping from numeric labels to class names
        scaler: Optional scaler for numeric features
        
    Returns:
        str: Predicted outcome category and class probabilities
    """
    # Create DataFrame with the same structure as the training data
    input_df = pd.DataFrame([user_input_features])
    
    # Apply same preprocessing as during training
    
    # Handle categorical features (if any)
    # Only process categorical columns that exist in user input
    for col in input_df.columns:
        if col in X_selected.columns and X_selected[col].dtype == 'object':
            # If value not in training data, use most common value
            print(f"Warning: Categorical feature '{col}' may require encoding.")
            input_df[col] = 0  # Default to most common class
    
    # Handle numeric features - apply the same scaling
    if scaler is not None:
        numeric_cols = [col for col in input_df.columns if col in X_selected.columns]
        if len(numeric_cols) > 0:
            try:
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            except:
                print("Warning: Could not apply scaling to input features. Using raw values.")
    
    # Ensure we only use the same features used during training
    # If we're missing columns from X_selected, add them with default values
    for col in X_selected.columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default to 0 for missing features
    
    # Reorder columns to match X_selected
    final_input = pd.DataFrame(columns=X_selected.columns)
    for col in X_selected.columns:
        if col in input_df.columns:
            final_input[col] = input_df[col]
        else:
            final_input[col] = 0
    
    # Make prediction
    try:
        pred_encoded = model.predict(final_input)[0]
        # Convert the prediction back to the original class label
        predicted_category = target_mapping.get(pred_encoded, "Unknown")
        
        # Also get prediction probabilities for all classes
        pred_probs = model.predict_proba(final_input)[0]
        class_probs = {target_mapping.get(i, f"Class {i}"): prob for i, prob in enumerate(pred_probs)}
        
        return predicted_category, class_probs
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", {}

def get_user_input(feature_columns):
    """
    Prompt the user for input values for each feature needed for prediction.
    
    Args:
        feature_columns: List of feature names required by the model
        
    Returns:
        dict: Dictionary of feature values entered by user
    """
    print("\nPlease enter the following information to predict the call outcome:")
    print("(Press Enter to use default values shown in brackets)\n")
    
    user_input = {}
    
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
    
    # For each feature in feature_columns, ask the user for input
    for feature in feature_columns:
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

def main():
    """Main prediction interface"""
    print(BANNER)
    
    # Check if the model file exists
    if not os.path.exists('saved_models/call_outcome_model.pkl'):
        print("Error: Model file not found.")
        print("Please run the training script (5.py) first to create the model.")
        return
    
    # Load the model and metadata
    try:
        print("Loading model...")
        model = joblib.load('saved_models/call_outcome_model.pkl')
        model_metadata = joblib.load('saved_models/call_outcome_metadata.pkl')
        
        feature_columns = model_metadata['feature_columns']
        target_mapping = model_metadata['target_mapping']
        scaler = model_metadata.get('scaler')
        
        print("Model loaded successfully!")
        
        # Create a dummy X_selected DataFrame with the correct feature columns
        X_selected = pd.DataFrame(columns=feature_columns)
        
        # Run the prediction loop
        while True:
            # Get user input
            user_input = get_user_input(feature_columns)
            
            # Make prediction
            outcome, probabilities = predict_call_outcome(
                user_input, model, X_selected, target_mapping, scaler
            )
            
            # Display results
            display_prediction(outcome, probabilities)
            
            # Ask if the user wants to make another prediction
            another = input("\nWould you like to predict another call outcome? (yes/no): ")
            if another.lower() not in ['yes', 'y']:
                print("\nThank you for using the Call Outcome Prediction Tool!")
                break
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model files exist and are not corrupted.")

if __name__ == "__main__":
    main() 