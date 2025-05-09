# ==============================================================================
#Kaggle Resource: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules for text processing, model selection, and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

# ---------------------
# STEP 1: Data Loading & Preprocessing
# ---------------------
# Load the dataset (make sure "Tweets.csv" is in your working directory)
tweets_df = pd.read_csv("Tweets.csv")

# Print basic information to understand dataset structure
print("Dataset shape:", tweets_df.shape)
print(tweets_df.head())

# Drop rows with missing values in key columns: 'text' and 'airline_sentiment'
tweets_df = tweets_df.dropna(subset=['text', 'airline_sentiment'])

# ---------------------
# STEP 2: Target Encoding
# ---------------------
# Use LabelEncoder to convert sentiment labels (e.g., negative, neutral, positive) into numeric values.
le_tweets = LabelEncoder()
tweets_df['sentiment_encoded'] = le_tweets.fit_transform(tweets_df['airline_sentiment'])

# Print mapping for clarity: shows how original sentiment labels map to numbers.
sentiment_mapping = dict(zip(le_tweets.classes_, le_tweets.transform(le_tweets.classes_)))
print("Twitter Sentiment Mapping:", sentiment_mapping)

# ---------------------
# STEP 3: Feature Engineering with TF-IDF Vectorization
# ---------------------
# KEY IMPROVEMENT:
# - Use bi-grams by setting ngram_range=(1,2) to capture word pairs.
# - Filter out infrequent terms (min_df=5) to reduce noise.
# - Use max_df=0.8 to ignore terms that appear in over 80% of documents.
tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2), min_df=5)
X_tweets = tfidf.fit_transform(tweets_df['text'])
y_tweets = tweets_df['sentiment_encoded']

# ---------------------
# STEP 4: Data Splitting
# ---------------------
# Split the data into training (80%) and testing (20%) sets for robust evaluation.
X_train, X_test, y_train, y_test = train_test_split(X_tweets, y_tweets, test_size=0.2, random_state=42)

# ---------------------
# STEP 5: Hyperparameter Tuning using GridSearchCV
# ---------------------
# KEY IMPROVEMENT:
# - Use GridSearchCV to tune the 'alpha' parameter of MultinomialNB (controls smoothing).
# - Good smoothing (alpha) helps reduce overfitting and improves recall & F1-score.
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
nb_model = MultinomialNB()
grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best alpha for Tweets model:", grid_search.best_params_)

# ---------------------
# STEP 6: Model Evaluation
# ---------------------
# Predict sentiments on the test set using the best model from GridSearchCV.
y_pred = best_model.predict(X_test)

# Calculate and print evaluation metrics to assess model reliability.
print("\n=== Twitter Airline Sentiment Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_tweets.classes_))

# ---------------------
# STEP 7: Visualization - Confusion Matrix
# ---------------------
# Plot the confusion matrix to visually assess how well the model predicts each sentiment.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_tweets.classes_, yticklabels=le_tweets.classes_)
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix for Twitter Airline Sentiment Model")
plt.show()

# ---------------------
# STEP 8: New Data Entry Provision for Prediction
# ---------------------
def predict_tweet_sentiment(new_tweet):
    """
    Predict the sentiment of a new tweet.
    
    Parameters:
      new_tweet (str): The tweet text to analyze.
      
    Returns:
      str: The predicted sentiment label (e.g., negative, neutral, positive).
      
    Process:
      1. Convert the new tweet into TF-IDF features using the fitted vectorizer.
      2. Predict the sentiment using the tuned Multinomial Naïve Bayes model.
      3. Transform the numeric prediction back to the original sentiment label.
    """
    # Transform new tweet text to TF-IDF features
    tweet_vector = tfidf.transform([new_tweet])
    # Predict the encoded sentiment value
    pred_encoded = best_model.predict(tweet_vector)[0]
    # Convert the numeric prediction back to the original label
    return le_tweets.inverse_transform([pred_encoded])[0]

# Example: Predict sentiment for a new tweet
sample_tweet = "I absolutely love the friendly service provided by this airline!"
predicted_sentiment = predict_tweet_sentiment(sample_tweet)
print("\nNew Tweet Prediction:", predicted_sentiment)

# ==============================================================================
# Explanation:
# ==============================================================================
# 1. Data Loading & Preprocessing:
#    - We load the "Tweets.csv" file and drop rows with missing 'text' or 'airline_sentiment'.
#
# 2. Target Encoding:
#    - The sentiment labels are encoded into numerical values using LabelEncoder.
#
# 3. Feature Engineering:
#    - TF-IDF vectorization converts the tweet text into a numerical format.
#    - Using bi-grams and filtering infrequent terms enhances feature quality.
#
# 4. Data Splitting:
#    - The data is split into training and testing sets for reliable model evaluation.
#
# 5. Hyperparameter Tuning:
#    - GridSearchCV tunes the 'alpha' parameter for the Multinomial Naïve Bayes model.
#    - Optimal alpha helps achieve higher accuracy, recall, and F1-score.
#
# 6. Model Evaluation:
#    - We compute accuracy, precision, recall, F1-score, and print a classification report.
#    - The confusion matrix graph visually represents model performance.
#
# 7. New Data Provision:
#    - The predict_tweet_sentiment function allows for real-time predictions on new tweet inputs.
#
# With these improvements and extensive feature engineering, the model should achieve an accuracy of 
# 70% or higher along with reliable recall and F1-scores.

# ==============================================================================
# STEP 9: User-Friendly Prediction Interface
# ==============================================================================
def run_sentiment_prediction_interface():
    """
    Provides a user-friendly interface for predicting sentiment of airline tweets.
    This function doesn't modify any existing model or data - it just provides
    a clean interface for making predictions.
    """
    print("\n" + "="*70)
    print("     AIRLINE TWEET SENTIMENT PREDICTION SYSTEM")
    print("="*70)
    
    print("\nThis tool analyzes airline-related tweets and predicts whether the")
    print("sentiment is positive, negative, or neutral.")
    print("\nThe model was trained on real Twitter data about airlines.")
    
    # Instructions for the user
    print("\nINSTRUCTIONS:")
    print("  - Enter the tweet text you want to analyze")
    print("  - Type 'exit' to quit the program")
    print("  - For best results, write as if you were tweeting about an airline")
    
    # Examples to help users understand
    print("\nEXAMPLES:")
    print("  - \"The flight was delayed by 3 hours and no explanation was given\"")
    print("  - \"Thank you for the excellent customer service and on-time arrival\"")
    print("  - \"Average flight experience, nothing special to report\"")
    
    # Main prediction loop
    while True:
        print("\n" + "-"*70)
        user_input = input("\nEnter a tweet to analyze (or 'exit' to quit): ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Airline Tweet Sentiment Analyzer!")
            break
        
        if not user_input.strip():
            print("Please enter a valid tweet text.")
            continue
        
        # Make prediction using existing function
        try:
            sentiment = predict_tweet_sentiment(user_input)
            
            # Display result with helpful explanation
            print("\nPREDICTION RESULT:")
            print("="*50)
            print(f"Sentiment: {sentiment.upper()}")
            
            # Add descriptive explanation based on sentiment
            if sentiment.lower() == 'positive':
                print("\nThis tweet expresses satisfaction or happiness with the airline service.")
                print("Examples of positive aspects might include: good customer service,")
                print("on-time performance, comfort, or helpful staff.")
            elif sentiment.lower() == 'negative':
                print("\nThis tweet expresses dissatisfaction or frustration with the airline service.")
                print("Examples of negative aspects might include: delays, cancellations,")
                print("poor customer service, lost baggage, or uncomfortable experience.")
            else:  # neutral
                print("\nThis tweet is neutral or factual without strong positive or negative emotions.")
                print("It may describe an experience without evaluative language or")
                print("contain both positive and negative elements that balance each other.")
            
            # Get prediction probabilities if available
            if hasattr(best_model, 'predict_proba'):
                try:
                    tweet_vector = tfidf.transform([user_input])
                    probabilities = best_model.predict_proba(tweet_vector)[0]
                    
                    # Display confidence scores
                    print("\nConfidence scores:")
                    for i, label in enumerate(le_tweets.classes_):
                        print(f"  {label.title()}: {probabilities[i]:.2%}")
                except:
                    pass
            
            print("\nReminder: This is an automated prediction and may not capture")
            print("all nuances of human communication.")
            
        except Exception as e:
            print(f"Error analyzing tweet: {e}")
            print("Please try again with different text.")

# Run the interface if executing this file directly
if __name__ == "__main__" and 'best_model' in globals() and 'tfidf' in globals() and 'le_tweets' in globals():
    # Check if we've already shown the example prediction
    sample_prediction_shown = True
    
    # Ask user if they want to try the interactive interface
    print("\nWould you like to try the interactive tweet sentiment analyzer?")
    user_choice = input("Enter 'yes' to continue or any other key to exit: ")
    
    if user_choice.lower() in ['yes', 'y', 'sure', 'ok']:
        run_sentiment_prediction_interface()
