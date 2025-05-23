## 1. Introduction

This project implements a custom AI-driven text classification system for sentiment analysis on synthetic customer feedback data. The system preprocesses text, trains a machine learning model (without using pre-trained models like BERT or libraries like Hugging Face), and provides a simple REST API endpoint for inference. The goal is to classify feedback into Positive, Negative, and Neutral categories.

This solution adheres to the requirements outlined in the assignment, including programmatic dataset generation, use of Python 3.8+ with allowed libraries (numpy, pandas, scikit-learn, flask), custom model implementation, and comprehensive documentation.

## 2. Approach and Design Decisions

### 2.1. Overall Pipeline

The system follows a standard machine learning pipeline:

1.  **Data Generation**: Programmatically create 1,000 synthetic customer feedback entries with balanced Positive, Negative, and Neutral labels.
2.  **Text Preprocessing**: Clean and normalize the text data by converting to lowercase, removing punctuation, tokenizing, removing stopwords, and performing lemmatization.
3.  **Feature Engineering**: Convert the preprocessed text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
4.  **Model Training & Selection**: Split the data into training (80%) and testing (20%) sets. Train and evaluate several scikit-learn classifiers (Logistic Regression, Multinomial Naive Bayes, SVM). The best-performing model is selected based on accuracy and other classification metrics.
5.  **API Implementation**: Develop a Flask-based REST API with a `/predict` endpoint that accepts text input and returns the predicted sentiment and confidence score.
6.  **Evaluation**: Report accuracy, precision, recall, and F1-score for each class on the test set.

### 2.2. Data Generation

-   **Method**: A Python script (`src/data_generator.py`) generates synthetic data using predefined templates for positive, negative, and neutral feedback. These templates include placeholders for product types and sentiment-specific keywords, which are randomly filled to create varied feedback entries.
-   **Rationale**: This approach ensures control over the dataset characteristics (length, sentiment balance) and avoids reliance on external datasets, as per the assignment constraints. It generates 1000 samples, approximately 333 for each sentiment class.

### 2.3. Text Preprocessing

-   **Steps** (`src/text_preprocessor.py`):
    1.  **Lowercasing**: Converts all text to lowercase for consistency.
    2.  **Punctuation Removal**: Removes all punctuation marks.
    3.  **Tokenization**: Splits text into individual words (tokens) using NLTK's `word_tokenize`.
    4.  **Stopword Removal**: Removes common English stopwords (e.g., "the", "is", "a") using NLTK's list.
    5.  **Lemmatization**: Reduces words to their base or dictionary form (lemma) using NLTK's `WordNetLemmatizer`. This helps in grouping different inflections of a word as a single feature.
-   **Rationale**: These steps are crucial for reducing noise, normalizing the text, and reducing the dimensionality of the feature space, leading to better model performance.

### 2.4. Feature Engineering

-   **Method**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the preprocessed text into a matrix of numerical features. `TfidfVectorizer` from scikit-learn is employed, with `max_features=5000` to limit the vocabulary size and prevent potential overfitting/high dimensionality with very rare words.
-   **Rationale**: TF-IDF is a standard and effective technique for text classification. It reflects how important a word is to a document in a collection or corpus, giving higher weight to terms that are frequent in a document but not across all documents.

### 2.5. Model Selection

-   **Models Evaluated**: Logistic Regression, Multinomial Naive Bayes, and Support Vector Machine (SVM with linear kernel and `probability=True` for confidence scores).
-   **Selection Criteria**: The model with the highest accuracy on the test set was chosen. Other metrics like precision, recall, and F1-score per class were also considered.
-   **Chosen Model**: **Logistic Regression** (solver=\'liblinear\', multi_class=\'auto\') was selected. On the generated synthetic dataset, all evaluated models achieved very high (near perfect) accuracy. Logistic Regression is a robust, interpretable, and computationally efficient model, making it a good choice for this task.
-   **Rationale**: The choice was driven by performance on the test set. Logistic Regression provides good baseline performance and is less prone to overfitting on smaller datasets compared to more complex models. The `liblinear` solver is suitable for binary and multiclass problems with L1 or L2 regularization.

### 2.6. API Design

-   **Framework**: Flask is used for its simplicity and lightweight nature, suitable for creating REST APIs quickly.
-   **Endpoint**: A `/predict` (POST) endpoint is provided. It accepts a JSON payload: `{"text": "your customer feedback here"}`.
-   **Response**: It returns a JSON response: `{"sentiment": "PredictedSentiment", "confidence": 0.XX}`.
-   **Error Handling**: Basic input validation (JSON format, presence/type of "text" field) is implemented.
-   **Caching (Bonus)**: An LRU (Least Recently Used) cache (`functools.lru_cache`) is implemented for the prediction function to store and quickly retrieve results for recently processed identical texts, improving response time for repeated queries.

## 3. Instructions to Run the Code and API

### 3.1. Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   NLTK resources (stopwords, punkt, wordnet, punkt_tab). The application attempts to download these if missing, but manual download might be required in some environments.

### 3.2. Setup

1.  **Clone/Download the Project**: Obtain the project files and navigate to the project root directory (`solulab_ai_assignment`).

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Resources (if not automatically handled or if issues persist)**:
    Open a Python interpreter and run:
    ```python
    import nltk
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("punkt_tab") # Added this based on runtime findings
    ```

### 3.3. Running the Model Training (Optional - Model is Pre-trained and Saved)

The best model pipeline is already trained and saved in the `models/` directory (`sentiment_model.joblib`). If you wish to retrain the model:

```bash
cd src
python sentiment_analyzer.py
```
This script will:
1.  Generate synthetic data.
2.  Preprocess the text.
3.  Train and evaluate Logistic Regression, Multinomial Naive Bayes, and SVM.
4.  Save the best performing model pipeline (vectorizer + classifier) to `models/sentiment_model.joblib`.

### 3.4. Running the Flask API Server

To start the API server:

```bash
cd src  # Ensure you are in the src directory where app.py is located
python app.py
```
The server will start, typically on `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/`.

### 3.5. Using the API

Once the server is running, you can send POST requests to the `/predict` endpoint. 

**Example using cURL:**

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"This product is absolutely amazing!\"}" http://127.0.0.1:5000/predict
```
OR 

```bash
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/predict" -Headers @{"Content-Type"="application/json"} -Body '{"text": "This product is absolutely amazing!"}' 
```

**Expected Response:**

```json
{
  "sentiment": "Positive",
  "confidence": 0.8051156207726192
}
```

**Health Check Endpoint:**

You can check the health of the API and if the model is loaded:

```bash
curl http://127.0.0.1:5000/health
```

**Expected Response:**
```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

## 4. Justification for Chosen Model and Features

-   **Model (Logistic Regression)**: As mentioned, Logistic Regression was chosen due to its strong performance on this synthetic dataset, its interpretability, and its efficiency. It provides a good balance between complexity and performance for text classification tasks, especially when features are well-engineered (like with TF-IDF). The `solver='liblinear'` is efficient for smaller datasets and supports L1/L2 regularization, which can help prevent overfitting. `multi_class='auto'` automatically selects the appropriate strategy (one-vs-rest or multinomial) based on the data and solver.
-   **Features (TF-IDF)**: TF-IDF was chosen because it's a proven and effective method for converting text into a meaningful numerical representation for machine learning models. It captures the importance of words in the context of the documents and the entire corpus. Using `max_features=5000` helps to keep the feature space manageable and focus on more relevant terms.
-   **Preprocessing**: The chosen preprocessing steps (lowercasing, punctuation removal, stopword removal, lemmatization) are standard practices that significantly improve model performance by reducing noise and standardizing the input text.

## 5. Assumptions and Limitations

### 5.1. Assumptions

-   **Synthetic Data**: The model is trained on synthetically generated data. Its performance on real-world, diverse customer feedback might vary and would likely require retraining or fine-tuning on actual domain-specific data.
-   **Sentiment Categories**: The sentiment is categorized into three distinct classes: Positive, Negative, and Neutral. Nuances like sarcasm, mixed sentiments within a single feedback, or domain-specific jargon might not be handled perfectly.
-   **Language**: The system is designed for English language text.
-   **Environment**: Assumes a standard Python environment where dependencies can be installed via pip.

### 5.2. Limitations

-   **No Deep Learning/Pre-trained Models**: As per assignment constraints, advanced models like BERT or other transformer-based architectures were not used. These models typically offer state-of-the-art performance on NLP tasks but require significantly more computational resources and data.
-   **Scalability of Training**: While the TF-IDF and Logistic Regression are relatively efficient, training on extremely large datasets (millions of entries) might require distributed computing solutions or more optimized data handling if done frequently.
The API itself is lightweight, but for very high throughput, a production-grade WSGI server (e.g., Gunicorn, uWSGI) and potentially load balancing would be needed.
-   **Confidence Scores**: The confidence scores from Logistic Regression (and SVM with `probability=True`) are probabilities. While useful, they are not always perfectly calibrated.
-   **Contextual Understanding**: The model relies on word occurrences (TF-IDF) and lacks deep contextual understanding that more complex models might capture.
-   **Out-of-Vocabulary Words**: Words not seen during training (and not within `max_features`) will be ignored by the TF-IDF vectorizer.

## 6. Bonus Points

### 6.1. API Caching

-   **Implementation**: An in-memory LRU (Least Recently Used) cache (`@functools.lru_cache(maxsize=128)`) has been added to the `get_prediction_with_cache` function in `src/app.py`. This caches the results of the last 128 unique prediction requests.
-   **Benefit**: For frequently repeated queries with the same input text, the API can return the cached result almost instantaneously, reducing redundant processing and improving response times.

### 6.2. Model Training Optimization (Proposal)

-   **Objective**: Make model training 20% faster without sacrificing accuracy.
-   **Proposed Optimization**: **Using `SGDClassifier` with `log_loss` for Logistic Regression-like behavior.**
    -   Scikit-learn's `SGDClassifier` allows for training linear models (including SVM, Logistic Regression) using Stochastic Gradient Descent. For large datasets, SGD is often much faster than solvers like `liblinear` because it processes samples one by one or in mini-batches.
    -   **Implementation Idea**: Replace `LogisticRegression(solver='liblinear')` with `SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)`. The `loss='log_loss'` makes `SGDClassifier` behave like a Logistic Regression model.
    -   **Why it could be faster**: SGD updates weights more frequently. While it might take more iterations to converge to the exact same optimum as batch methods, it often gets very close much faster, especially with larger datasets. For smaller datasets like the one here (1000 samples), the speedup might be less dramatic or even negligible compared to `liblinear`, but the principle applies for scalability.
    -   **Verification**: One would need to benchmark the training time of `SGDClassifier` against `LogisticRegression(solver='liblinear')` on this dataset and potentially larger ones to confirm the speedup while ensuring accuracy remains comparable (e.g., by tuning `max_iter`, `alpha` - regularization strength, and `learning_rate` schedule for `SGDClassifier`).
    -   **Alternative for TF-IDF**: If TF-IDF calculation is a bottleneck for very large datasets, using `HashingVectorizer` followed by `TfidfTransformer` can be more memory-efficient and sometimes faster, though it might lead to a slight decrease in accuracy due to hash collisions.

### 6.3. Cloud Deployment Discussion

Deploying this solution in a cloud environment (e.g., AWS, GCP, Azure) would involve several considerations for scalability, reliability, and maintainability:

1.  **Containerization (Docker)**:
    -   Package the Flask application, its dependencies, and the trained model into a Docker container. This ensures consistency across different environments.
    -   A `Dockerfile` would define the image, including installing Python, copying application code, installing `requirements.txt`, and specifying the command to run the Flask app (e.g., using Gunicorn).

2.  **Compute Service**:
    -   **AWS**: AWS Elastic Beanstalk, AWS App Runner, or Amazon ECS/EKS (Elastic Container Service/Kubernetes Service) for deploying the Docker container. Elastic Beanstalk is simpler for web apps, while ECS/EKS offer more control.
    -   **GCP**: Google App Engine, Cloud Run, or Google Kubernetes Engine (GKE). Cloud Run is excellent for stateless containerized applications and scales to zero.
    -   **Azure**: Azure App Service, Azure Container Instances, or Azure Kubernetes Service (AKS).

3.  **API Gateway**:
    -   Place an API Gateway (e.g., Amazon API Gateway, Google Cloud API Gateway, Azure API Management) in front of the application. This provides features like request throttling, caching (at the gateway level), authentication/authorization, monitoring, and traffic management.

4.  **Model Management & Storage**:
    -   Store the trained model file (`sentiment_model.joblib`) in a cloud storage service (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage).
    -   The application container would download the model from this storage upon startup.
    -   For more advanced MLOps, use services like AWS SageMaker Model Registry, Google Vertex AI Model Registry, or Azure Machine Learning Model Management to version and manage models.

5.  **Scalability & High Availability**:
    -   Configure auto-scaling for the compute service based on metrics like CPU utilization or request count. This ensures the application can handle varying loads.
    -   Deploy the application across multiple Availability Zones (AZs) for high availability.

6.  **Logging and Monitoring**:
    -   Integrate with cloud-native logging and monitoring services (e.g., AWS CloudWatch, Google Cloud Logging/Monitoring, Azure Monitor) to track application performance, errors, and API usage.

7.  **CI/CD Pipeline**:
    -   Set up a CI/CD pipeline (e.g., AWS CodePipeline, Jenkins, GitLab CI, GitHub Actions) to automate testing, building the Docker image, and deploying updates to the cloud environment.

8.  **Retraining Pipeline (for production systems)**:
    -   For a real-world system, a pipeline to periodically retrain the model on new data would be essential. This could be orchestrated using services like AWS Step Functions, Google Cloud Composer (Airflow), or Azure Data Factory.
    -   The new data would be collected, preprocessed, and used to train a new model version. After evaluation, the new model could be promoted to production.

**Example: AWS Deployment using Elastic Beanstalk**

1.  Create a `Dockerfile`.
2.  Zip the application code (including `Dockerfile`, `src/`, `models/`, `requirements.txt`).
3.  Create an Elastic Beanstalk application and environment (Python platform).
4.  Upload the ZIP file. Elastic Beanstalk would build the Docker image (if `Dockerfile` is present and platform is Docker) or set up the Python environment and deploy the application.
5.  Configure environment variables, instance types, and auto-scaling rules.

This cloud setup would provide a scalable, robust, and maintainable deployment for the sentiment analysis API.

## 7. Project Structure

```
solulab_ai_assignment/
├── models/
│   └── sentiment_model.joblib  # Saved trained model pipeline
├── src/
│   ├── __init__.py
│   ├── app.py                  # Flask API implementation
│   ├── data_generator.py       # Synthetic data generation script
│   ├── sentiment_analyzer.py   # Model training and evaluation pipeline
│   └── text_preprocessor.py    # Text preprocessing functions
├── requirements.txt            # Python dependencies
├── todo.md                     # Task checklist (internal use)
└── README.md                   # This file
```


