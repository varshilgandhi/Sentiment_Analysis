import re
import string
# It's good practice to download nltk resources if not present, 
# but for this environment, we assume they are available or handle it in main script setup.
# import nltk
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available (this might be better in a setup script or main)
# try:
#     stopwords.words('english')
#     word_tokenize("test")
#     WordNetLemmatizer().lemmatize("tests")
# except LookupError:
#     print("NLTK resources not found. Downloading...")
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     print("NLTK resources downloaded.")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    - Lowercases
    - Removes punctuation
    - Removes numbers (optional, kept for now as numbers might be relevant in some feedback)
    - Tokenizes
    - Removes stopwords
    - Lemmatizes
    """
    if not isinstance(text, str):
        return "" # Or raise an error, or return a specific marker for missing text

    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers (optional - decide based on dataset and problem)
    # For now, let's keep numbers as they might be part of feedback, e.g., "item broke after 1 day"
    # text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()
    ] # also ensuring tokens are alphabetic after punctuation/number removal
    
    return " ".join(processed_tokens)

if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "This is a FANTASTIC product! Loved it 100%.",
        "The item was terrible, broke after 2 days. Very disappointing experience.",
        "It's an okay product, nothing special to write home about.",
        None, # Test with None
        "   ", # Test with empty string
        "12345 numeric only"
    ]
    
    print("Original vs. Processed Text:")
    for text in sample_texts:
        print(f"Original: {text}")
        print(f"Processed: {preprocess_text(text)}\n")

    # Test with a pandas Series (common use case)
    import pandas as pd
    data = {'feedback': ["Great service, very happy!", "Awful, will not buy again.", "It is average."]}
    df = pd.DataFrame(data)
    df['processed_feedback'] = df['feedback'].apply(preprocess_text)
    print("\nPandas DataFrame processing example:")
    print(df)

