# Fake News Detection System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 1: Load datasets
# -------------------------------
fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

# Add labels
fake_data["label"] = 0   # fake = 0
true_data["label"] = 1   # real = 1

# Combine both datasets
data = pd.concat([fake_data, true_data], axis=0)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# -------------------------------
# Step 2: Select useful column
# -------------------------------
# We will use 'text' column for prediction
data = data[["text", "label"]]

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Convert text to numbers
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Step 5: Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -------------------------------
# Step 6: Test Model
# -------------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -------------------------------
# Step 7: Custom Prediction
# -------------------------------
def predict_news(news_text):
    news_vec = vectorizer.transform([news_text])
    prediction = model.predict(news_vec)

    if prediction[0] == 0:
        print("This news is FAKE ❌")
    else:
        print("This news is REAL ✅")


# -------------------------------
# Step 8: Try your own input
# -------------------------------
print("\nEnter a news statement:")
user_input = input()

predict_news(user_input)