import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. SETUP & CLEANING FUNCTION
def clean_text(text):
    """
    Standard NLP Cleaning Pipeline.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions/hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Fix extra whitespace
    return text

# 2. LOADING DATASETS
print("Loading datasets...")
train_bin = pd.read_csv("Binarylabel_Train.csv")
val_bin = pd.read_csv("Binarylabel_Val.csv")
test_bin = pd.read_csv("Binarylabel_Test - Competition.csv")

train_multi = pd.read_csv("Multilabel_Train.csv")
val_multi = pd.read_csv("Multilabel_Val.csv")
test_multi = pd.read_csv("Multilabel_Test - Competition.csv")

# 3. PREPROCESSING
print("Preprocessing text data...")
for df in [train_bin, val_bin, test_bin, train_multi, val_multi, test_multi]:
    df['clean_text'] = df['text'].apply(clean_text)

# 4. VECTORIZATION (TF-IDF)
# Converting text to numerical features.
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

# 5. TASK A: BINARY CLASSIFICATION (Champion: Logistic Regression)
print("Training Binary Model (Logistic Regression)...")
X_train_bin = tfidf.fit_transform(train_bin['clean_text'])
X_val_bin = tfidf.transform(val_bin['clean_text'])
X_test_bin = tfidf.transform(test_bin['clean_text'])

# Using 'balanced' weights to handle 79/21 class imbalance.
model_bin = LogisticRegression(class_weight='balanced', max_iter=1000)
model_bin.fit(X_train_bin, train_bin['label'])

# Validation Results
y_pred_bin = model_bin.predict(X_val_bin)
print("\n--- Binary Task Validation Results ---")
print(f"Accuracy: {accuracy_score(val_bin['label'], y_pred_bin):.4f}")
print(f"Weighted F1-Score: {f1_score(val_bin['label'], y_pred_bin, average='weighted'):.4f}")

# 6. TASK B: MULTICLASS CLASSIFICATION (Champion: Linear SVM)
print("\nTraining Multiclass Model (Linear SVM)...")
X_train_multi = tfidf.fit_transform(train_multi['clean_text'])
X_val_multi = tfidf.transform(val_multi['clean_text'])
X_test_multi = tfidf.transform(test_multi['clean_text'])

# Handling 73% 'None' class imbalance.
model_multi = LinearSVC(class_weight='balanced', max_iter=10000)
model_multi.fit(X_train_multi, train_multi['label'])

# Validation Results
y_pred_multi = model_multi.predict(X_val_multi)
print("\n--- Multiclass Task Validation Results ---")
print(f"Accuracy: {accuracy_score(val_multi['label'], y_pred_multi):.4f}")
print(f"Weighted F1-Score: {f1_score(val_multi['label'], y_pred_multi, average='weighted'):.4f}")

# 7. GENERATING COMPETITION PREDICTIONS
print("\nGenerating final competition predictions...")
test_bin['label'] = model_bin.predict(X_test_bin)
test_multi['label'] = model_multi.predict(X_test_multi)

# Saving final CSV files
test_bin[['unique_id', 'text', 'label']].to_csv("Binarylabel_Test_Submission.csv", index=False)
test_multi[['unique_id', 'text', 'label']].to_csv("Multilabel_Test_Submission.csv", index=False)

print("\nSUCCESS: All tasks complete. Submission files generated.")
