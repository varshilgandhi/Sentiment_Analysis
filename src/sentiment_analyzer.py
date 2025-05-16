import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib # For saving the model and vectorizer
import os

# Import custom modules
from data_generator import generate_dataset
from text_preprocessor import preprocess_text

# Define project root for saving models/data if necessary
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

def train_and_evaluate():
    """
    Orchestrates the data generation, preprocessing, model training, and evaluation.
    """
    print("Starting the sentiment analysis pipeline...")

    # 1. Data Generation
    print("\nStep 1: Generating synthetic dataset...")
    df = generate_dataset(num_samples=1000)
    print(f"Dataset generated with {len(df)} samples.")
    print(df["sentiment"].value_counts())

    # 2. Text Preprocessing
    print("\nStep 2: Preprocessing text data...")
    # Ensure NLTK resources are available (handled by prior steps, but good to be aware)
    # nltk.download('stopwords', quiet=True)
    # nltk.download('punkt', quiet=True)
    # nltk.download('wordnet', quiet=True)
    df["processed_text"] = df["text"].apply(preprocess_text)
    print("Text preprocessing complete.")
    print(df[["text", "processed_text"]].head())

    # Filter out any empty strings that might result from preprocessing if text was only stopwords/punctuation
    df = df[df["processed_text"].str.strip().astype(bool)]
    print(f"Dataset size after removing empty processed text: {len(df)}")

    # 3. Feature Engineering (TF-IDF)
    print("\nStep 3: Performing TF-IDF Vectorization...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting features can help with performance and overfitting

    # 4. Data Splitting
    print("\nStep 4: Splitting data into training and testing sets (80/20)...")
    X = df["processed_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Define models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', multi_class='auto', random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42) # probability=True for confidence scores
    }

    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    trained_pipeline = None

    print("\nStep 5 & 6: Training and Evaluating Models...")
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        # Create a pipeline with TF-IDF and the classifier
        pipeline = Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("classifier", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model # We save the pipeline later
            best_model_name = name
            trained_pipeline = pipeline # Save the entire pipeline

    print(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # Save the best model pipeline (vectorizer + classifier)
    if trained_pipeline:
        print(f"\nSaving the best model pipeline ({best_model_name}) to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")
        # The pipeline already contains the fitted vectorizer as its first step
        joblib.dump(trained_pipeline, MODEL_PATH)
        # We can also save the vectorizer separately if needed for other purposes, but it's part of the pipeline.
        # joblib.dump(trained_pipeline.named_steps['tfidf'], VECTORIZER_PATH)
        print("Model pipeline saved successfully.")
    else:
        print("No model was trained successfully to save.")

    return trained_pipeline, X_test, y_test # Return for further use if needed, e.g. API

if __name__ == "__main__":
    # This will run the full training and evaluation pipeline
    # The Flask API part will be in a separate script or integrated later.
    final_pipeline, _, _ = train_and_evaluate()
    if final_pipeline:
        print("\nSentiment analysis model training and evaluation complete.")
        print(f"The best model is: {final_pipeline.named_steps['classifier']}")

