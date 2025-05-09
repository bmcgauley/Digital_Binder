import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the IDS Dataset
print("Loading IDS Dataset...")

try:
    df = pd.read_csv('data.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print("Error: File 'data.csv' not found.")
    exit()

# Print initial information about the dataset
print("\nInitial dataset information:")
print(df.info())
print("\nSample of the data:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('Number of Barriers', axis=1)  # Using Number of Barriers as target
y = df['Number of Barriers']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find best k using cross-validation
print("\nFinding optimal k...")
k_values = [2, 3, 4, 5, 7, 9, 13, 15]
best_k = None
best_score = float('-inf')

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    mean_score = scores.mean()
    print(f"k={k}, R² Score={mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nBest k: {best_k} (R² Score: {best_score:.4f})")

# Train final model
print("\nTraining final model...")
final_model = KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = final_model.predict(X_test_scaled)

print("\nTest Set Evaluation:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

def predict_barriers(input_data):
    """
    Make a prediction using the trained KNN model.
    """
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = final_model.predict(input_scaled)
    
    return prediction[0]

def get_user_input():
    """
    Get user input for making predictions.
    """
    print("\nIntrusion Detection System (IDS) Configuration Parameters")
    print("======================================================")
    print("\nThis system will help you determine the optimal number of security barriers")
    print("needed for your wireless sensor network based on your deployment parameters.")
    print("\nPlease provide the following information:")
    
    user_input = {}
    
    # Area input
    print("\n1. Area Configuration:")
    print("   The total area of your deployment zone in square meters.")
    print("   Typical range: 1000-10000 square meters")
    print("   Larger areas typically require more barriers for effective coverage.")
    while True:
        try:
            value = float(input("\nEnter Area (in square meters): "))
            if value <= 0:
                print("Area must be positive")
                continue
            if value < 1000:
                print("Warning: Area seems small. This might affect sensor coverage.")
            elif value > 10000:
                print("Warning: Large area. May require significant number of barriers.")
            user_input['Area'] = value
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Sensing Range input
    print("\n2. Sensing Range Configuration:")
    print("   The radius within which each sensor can detect intrusions.")
    print("   Typical range: 10-50 meters")
    print("   Larger sensing ranges provide better coverage but may consume more power.")
    while True:
        try:
            value = float(input("\nEnter Sensing Range (in meters): "))
            if value <= 0:
                print("Sensing Range must be positive")
                continue
            if value < 10:
                print("Warning: Small sensing range may create coverage gaps.")
            elif value > 50:
                print("Warning: Large sensing range may impact sensor battery life.")
            user_input['Sensing Range'] = value
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Transmission Range input
    print("\n3. Transmission Range Configuration:")
    print("   The maximum distance at which sensors can communicate with each other.")
    print("   Typical range: 20-100 meters")
    print("   Should be greater than sensing range for reliable communication.")
    while True:
        try:
            value = float(input("\nEnter Transmission Range (in meters): "))
            if value <= 0:
                print("Transmission Range must be positive")
                continue
            if value <= user_input['Sensing Range']:
                print("Warning: Transmission range should typically be larger than sensing range.")
            user_input['Transmission Range'] = value
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Number of Sensor nodes input
    print("\n4. Number of Sensor Nodes:")
    print("   Total number of sensor devices to be deployed in the network.")
    print("   More nodes generally provide better coverage but increase cost.")
    print("   Recommended: At least 1 node per 100 square meters for basic coverage.")
    while True:
        try:
            value = float(input("\nEnter Number of Sensor nodes: "))
            if value <= 0:
                print("Number of nodes must be positive")
                continue
            recommended_min = user_input['Area'] / 100
            if value < recommended_min:
                print(f"Warning: Number of nodes may be too low for the area.")
                print(f"Recommended minimum: {recommended_min:.0f} nodes for your area.")
            user_input['Number of Sensor nodes'] = value
            break
        except ValueError:
            print("Please enter a valid number")
    
    return user_input

def interactive_prediction():
    """
    Run interactive prediction loop.
    """
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            # Make prediction
            predicted_barriers = predict_barriers(user_input)
            
            # Print results
            print("\nIDS Configuration Analysis")
            print("========================")
            
            # Coverage Analysis
            area = user_input['Area']
            sensing_range = user_input['Sensing Range']
            nodes = user_input['Number of Sensor nodes']
            transmission_range = user_input['Transmission Range']
            
            # Calculate key metrics
            theoretical_coverage_per_node = np.pi * sensing_range**2
            total_theoretical_coverage = theoretical_coverage_per_node * nodes
            coverage_ratio = (total_theoretical_coverage / area) * 100
            
            print("\n1. Deployment Parameters:")
            print(f"   • Area Coverage: {area:,.0f} square meters")
            print(f"   • Sensing Range: {sensing_range:.1f} meters")
            print(f"   • Transmission Range: {transmission_range:.1f} meters")
            print(f"   • Number of Sensor Nodes: {nodes:.0f}")
            
            print("\n2. Coverage Analysis:")
            print(f"   • Coverage per node: {theoretical_coverage_per_node:.1f} square meters")
            print(f"   • Total theoretical coverage: {total_theoretical_coverage:,.1f} square meters")
            print(f"   • Coverage ratio: {coverage_ratio:.1f}%")
            
            if coverage_ratio < 100:
                print("   ⚠ Warning: Potential coverage gaps in the network")
            elif coverage_ratio > 200:
                print("   ℹ Note: High sensor overlap, might be over-provisioned")
            
            print("\n3. Barrier Prediction:")
            print(f"   • Predicted Number of Barriers: {predicted_barriers:.0f}")
            print("\n   What this means:")
            print(f"   • The model suggests deploying {predicted_barriers:.0f} security barriers")
            print("   • Each barrier represents a line of defense against intrusion")
            print("   • More barriers generally mean:")
            print("     - Higher security level")
            print("     - Better intrusion detection probability")
            print("     - More redundancy in case of node failures")
            
            print("\n4. Network Recommendations:")
            if transmission_range < 2 * sensing_range:
                print("   ⚠ Consider increasing transmission range for better connectivity")
            if coverage_ratio < 90:
                print("   ⚠ Consider adding more sensor nodes to improve coverage")
            if predicted_barriers < 10:
                print("   ⚠ Low number of barriers might indicate security vulnerabilities")
            elif predicted_barriers > 100:
                print("   ℹ High number of barriers might indicate over-provisioning")
            
            # Ask if user wants to make another prediction
            print("\n==============================================")
            again = input("Would you like to analyze another configuration? (yes/no): ").lower()
            if again != 'yes':
                print("\nThank you for using the IDS Configuration Analyzer!")
                break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again")

# Run the interactive prediction system
if __name__ == "__main__":
    print("\nWelcome to the IDS Configuration Analyzer")
    print("=======================================")
    print("This system helps you determine the optimal number of security barriers")
    print("needed for your Intrusion Detection System based on your deployment parameters.")
    print("\nModel Performance Summary:")
    print(f"Best k value: {best_k}")
    print(f"Cross-validation R² Score: {best_score:.4f}")
    print("\nStarting configuration analysis...")
    interactive_prediction() 