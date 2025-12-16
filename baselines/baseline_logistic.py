import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Parameters
DATA_FILE = "multiclass-bias.csv"  # Use balanced set
MODEL_NAME = "logistic_regression_classifier"

print("Training Logistic Regression Multiclass Bias Classifier...")

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_path, DATA_FILE))
print(f"Loaded {len(df)} samples from {DATA_FILE}")

# Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['bias_label'])
print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")

# Train/test split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")

# TF-IDF Vectorization
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
X_train = vectorizer.fit_transform(train_df['text'].astype(str))
X_test = vectorizer.transform(test_df['text'].astype(str))
y_train = train_df['label_id'].values
y_test = test_df['label_id'].values

print(f"Feature matrix shape: {X_train.shape}")

# Train Logistic Regression
print("\nTraining Logistic Regression model...")
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

# Check number of iterations
print(f"Number of iterations: {model.n_iter_[0] if hasattr(model.n_iter_, '__len__') else model.n_iter_}")

# Evaluate on test set
print("\nEvaluating on test set...")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Save model components
save_dir = os.path.join(base_path, MODEL_NAME)
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "model.joblib"))
joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.joblib"))
joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))

print(f"\nModel, vectorizer, and label encoder saved to {save_dir}")
