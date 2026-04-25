import pandas as pd
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Loading data...")

fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

fake_data["label"] = 0
true_data["label"] = 1

data = pd.concat([fake_data, true_data], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

data = data[["title", "label"]]

print("Training model...")

X = data["title"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model ready!")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

def show_metrics():
    metrics = ["Accuracy", "Precision", "Recall"]
    values = [accuracy, precision, recall]

    plt.figure()
    plt.bar(metrics, values)
    plt.title("Model Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()

def check_news():
    news = text_area.get("1.0", END).strip()

    if news == "":
        messagebox.showwarning("Warning", "Please enter some news text!")
        return

    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)

    if result[0] == 0:
        output_label.config(text="FAKE NEWS ❌", fg="red")
    else:
        output_label.config(text="REAL NEWS ✅", fg="green")

def clear_text():
    text_area.delete("1.0", END)
    output_label.config(text="")

root = Tk()
root.title("Fake News Detector")
root.geometry("600x550")
title_label = Label(root, text="Fake News Detection System", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

text_area = Text(root, height=10, width=60)
text_area.pack(pady=10)

check_button = Button(root, text="Check News", command=check_news, bg="blue", fg="white")
check_button.pack(pady=5)

clear_button = Button(root, text="Clear", command=clear_text)
clear_button.pack(pady=5)

metrics_button = Button(root, text="Show Metrics", command=show_metrics, bg="green", fg="white")
metrics_button.pack(pady=5)

output_label = Label(root, text="", font=("Arial", 14, "bold"))
output_label.pack(pady=20)

root.mainloop()