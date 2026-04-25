# Fake News Detection System
# Intermediate-level implementation

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# Text Cleaning Function
# ----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)         # remove special characters
    text = re.sub(r'\d', ' ', text)         # remove numbers
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    return text.strip()

# ----------------------------------
# Load and Prepare Data
# ----------------------------------
def load_data():
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], axis=0)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text (better accuracy)
    data["content"] = data["title"] + " " + data["text"]

    # Clean text
    data["content"] = data["content"].apply(clean_text)

    return data[["content", "label"]]

# ----------------------------------
# Train Model
# ----------------------------------
def train_model(data):
    X = data["content"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.75)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test_vec, y_test

# ----------------------------------
# Evaluate Model
# ----------------------------------
def evaluate(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# Predict Custom News
# ----------------------------------
def predict_news(model, vectorizer, text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)

    return "REAL" if prediction[0] == 1 else "FAKE"

# ----------------------------------
# Main Execution
# ----------------------------------
def main():
    print("Loading data...")
    data = load_data()

    print("Training model...")
    model, vectorizer, X_test_vec, y_test = train_model(data)

    print("Evaluating model...")
    evaluate(model, X_test_vec, y_test)

    # User input loop
    while True:
        print("\nEnter news text (or type 'exit'):")
        user_input = input()

        if user_input.lower() == "exit":
            break

        result = predict_news(model, vectorizer, user_input)
        print("Prediction:", result)


if __name__ == "__main__":
    main()