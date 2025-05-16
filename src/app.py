from flask import Flask, request, jsonify
import joblib
import os
import time
from functools import lru_cache

# Import custom modules (assuming they are in the same directory or PYTHONPATH is set)
# For a structured project, ensure src is on the path or use relative imports if running as a module
from text_preprocessor import preprocess_text # Make sure this can be imported

# Define project root and model paths
# This assumes app.py is in the src directory, and models is a sibling to src or at project root
# Adjust paths if your structure is different.
# For simplicity, let's assume models directory is at ../models relative to this script (src/app.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")

app = Flask(__name__)

# Load the trained model pipeline (TF-IDF vectorizer + classifier)
print(f"Loading model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at {MODEL_PATH}. Ensure the model is trained and saved.")
    model_pipeline = None # Or handle error appropriately

# Bonus: In-memory cache for recent predictions
@lru_cache(maxsize=128) # Cache up to 128 recent unique predictions
def get_prediction_with_cache(text_input):
    """Cached prediction function."""
    if not model_pipeline:
        return {"error": "Model not loaded"}, 500

    processed_text = preprocess_text(text_input)
    if not processed_text.strip(): # Handle cases where preprocessing results in empty string
        # Decide on a default sentiment or error for empty/non-alphabetic input
        return {"sentiment": "Neutral", "confidence": 0.5, "message": "Input resulted in empty text after preprocessing."}

    # The pipeline expects a list or iterable of texts
    prediction = model_pipeline.predict([processed_text])[0]
    
    # Get confidence score (probability)
    # Ensure the classifier in the pipeline supports predict_proba
    try:
        probabilities = model_pipeline.predict_proba([processed_text])[0]
        # Get the confidence of the predicted class
        class_index = list(model_pipeline.classes_).index(prediction)
        confidence = probabilities[class_index]
    except AttributeError:
        # If predict_proba is not available (e.g., for some SVM kernels without probability=True)
        confidence = "N/A (model does not support probability estimates)" 
    except Exception as e:
        confidence = f"Error getting confidence: {str(e)}"

    return {"sentiment": prediction, "confidence": confidence}

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "text" not in data or not isinstance(data["text"], str):
        return jsonify({"error": "Missing or invalid 'text' field in JSON payload"}), 400

    text_input = data["text"]
    if not text_input.strip():
        return jsonify({"error": "Input text cannot be empty"}), 400

    # Simulate some processing time for cache demonstration if needed
    # time.sleep(0.1) 

    result = get_prediction_with_cache(text_input)
    
    if "error" in result:
         return jsonify(result), result.get("status_code", 500)
         
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model_pipeline is not None}), 200

if __name__ == "__main__":
    # Ensure NLTK resources are downloaded before starting the app if text_preprocessor needs them
    # This should ideally be part of a setup/startup script
    try:
        import nltk
        from nltk.corpus import stopwords
        stopwords.words("english") # Check if stopwords are available
        nltk.word_tokenize("test") # Check if punkt is available
        nltk.stem.WordNetLemmatizer().lemmatize("test") # Check if wordnet is available
    except LookupError as e:
        print(f"NLTK resource missing: {e}. Please run the NLTK downloader script or ensure resources are present.")
        # Example: nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')
        # For this assignment, we assume they were downloaded by previous steps.
    
    print("Starting Flask API server...")
    # Make sure to run on 0.0.0.0 to be accessible outside the container/sandbox if needed for testing
    # The deploy_expose_port tool will handle public access.
    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False for production/assignment submission

