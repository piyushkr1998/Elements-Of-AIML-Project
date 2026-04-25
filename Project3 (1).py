# Fake News Detection with GUI (Tkinter)
# Simple and beginner-friendly project

import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Step 1: Load datasets
# -------------------------------
print("Loading data...")

fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

fake_data["label"] = 0
true_data["label"] = 1

data = pd.concat([fake_data, true_data], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

data = data[["title", "label"]]

# -------------------------------
# Step 2: Train model
# -------------------------------
print("Training model...")

X = data["title"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("Model ready!")

# -------------------------------
# Step 3: Prediction Function
# -------------------------------
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

# -------------------------------
# Step 4: Clear Function
# -------------------------------
def clear_text():
    text_area.delete("1.0", END)
    output_label.config(text="")

# -------------------------------
# Step 5: GUI Design
# -------------------------------
root = Tk()
root.title("Fake News Detector")
root.geometry("600x500")

# Heading
title_label = Label(root, text="Fake News Detection System", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Text box
text_area = Text(root, height=10, width=60)
text_area.pack(pady=10)

# Buttons
check_button = Button(root, text="Check News", command=check_news, bg="blue", fg="white")
check_button.pack(pady=5)

clear_button = Button(root, text="Clear", command=clear_text)
clear_button.pack(pady=5)

# Output label
output_label = Label(root, text="", font=("Arial", 14, "bold"))
output_label.pack(pady=20)

# Run GUI
root.mainloop()