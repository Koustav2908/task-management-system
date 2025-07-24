# Train the categorize model

import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

print("=" * 50)
print("PART 1: AI-POWERED TASK MANAGEMENT SYSTEM: TRAIN CATEGORIZATION MODEL")
print("=" * 50)

# Load dataset
df = pd.read_csv("tasks.csv")
print(f"Dataset loaded: {df.shape[0]} tasks, {df['category'].nunique()} categories")

# ==============================================================================
# NLP BASICS - TOKENIZATION, POS, LEMMATIZATION
# ==============================================================================

print("\n📚 NLP BASICS")
print("-" * 30)

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm"
    )

    # Fallback to simple cleaning if spaCy model is not available
    def clean_and_lemmatize(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return " ".join(text.split())

    nlp = None  # Set nlp to None if model not loaded


def clean_and_lemmatize(text):
    """
    Advanced text cleaning and lemmatization using spaCy.
    Removes stop words, punctuation, and converts words to their base form.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Replace non-alphabetic with space

    if nlp:  # Use spaCy only if the model was loaded successfully
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
        ]
        return " ".join(tokens)
    else:  # Fallback to simple cleaning if spaCy model is not available
        return " ".join(text.split())


# Clean and combine text using the enhanced function
df["title_desc"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(
    clean_and_lemmatize
)

print("✅ Text preprocessing completed")
print(f"Sample cleaned text: {df['title_desc'].iloc[0]}")

# Simple tokenization example (using the new cleaned text)
sample_text = df["title_desc"].iloc[0]
tokens = sample_text.split()
print(f"Tokens: {tokens[:5]}...")

# ==============================================================================
# TEXT VECTORIZATION - TF-IDF
# ==============================================================================

print("\n📊 TEXT VECTORIZATION")
print("-" * 30)

# TF-IDF Vectorization with increased max_features
tfidf = TfidfVectorizer(
    max_features=1500,
    stop_words="english",  # This might be redundant if spaCy already removed them, but good as a safeguard
    ngram_range=(1, 2),
    min_df=2,
)

# Transform text to vectors
X = tfidf.fit_transform(df["title_desc"])  # Use the new 'title_desc' column
print(f"✅ TF-IDF matrix created: {X.shape}")

# Show top features by category
print("\n🔍 Top terms by category:")
for category in df["category"].unique()[:3]:  # Show only 3 categories
    category_data = df[df["category"] == category][
        "title_desc"
    ]  # Use the new 'title_desc' column
    category_tfidf = tfidf.transform(category_data)
    mean_scores = np.array(category_tfidf.mean(axis=0)).flatten()
    top_words = [
        tfidf.get_feature_names_out()[i] for i in mean_scores.argsort()[-3:][::-1]
    ]
    print(f"{category}: {top_words}")

# ==============================================================================
# ML MODELS FOR TASK CATEGORIZATION
# ==============================================================================

print("\n🤖 ML MODELS")
print("-" * 30)

# Prepare data
y_cat = df["category"]
y_pri = df["priority"]

category_le = LabelEncoder()
y_category = category_le.fit_transform(y_cat)

priority_le = LabelEncoder()
y_priority = priority_le.fit_transform(y_pri)

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_category, test_size=0.2, random_state=42, stratify=y_category
)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)

print(
    f"Category Model: Training: {X_train_c.shape[0]} samples, Testing: {X_test_c.shape[0]} samples"
)
print(
    f"Priority Model: Training: {X_train_p.shape[0]} samples, Testing: {X_test_p.shape[0]} samples"
)

# Model 1: Logistic Regression with GridSearchCV
print("\n📊 Logistic Regression with GridSearchCV:")
# Define parameter grid for LogisticRegression - Expanded C range
lr_param_grid = {
    "C": [0.1, 1, 10, 100],  # Expanded Regularization strength
    "solver": ["liblinear", "lbfgs"],  # Different solvers
    "max_iter": [1000, 2000, 3000],  # Expanded max_iter
}

# Initialize GridSearchCV for Logistic Regression (Category)
grid_search_lr_c = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,  # Use all available cores
    verbose=1,
)
grid_search_lr_c.fit(X_train_c, y_train_c)

# Initialize GridSearchCV for Logistic Regression (Priority)
grid_search_lr_p = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,  # Use all available cores
    verbose=1,
)
grid_search_lr_p.fit(X_train_p, y_train_p)

# Get the best Logistic Regression model and its accuracy (Category)
lr_c = grid_search_lr_c.best_estimator_
lr_pred_c = lr_c.predict(X_test_c)
lr_acc_c = accuracy_score(y_test_c, lr_pred_c)
print("Category Model")
print(f"Best Logistic Regression Parameters: {grid_search_lr_c.best_params_}")
print(f"Logistic Regression Test Accuracy (tuned): {lr_acc_c:.3f}")

# Get the best Logistic Regression model and its accuracy (Priority)
lr_p = grid_search_lr_p.best_estimator_
lr_pred_p = lr_c.predict(X_test_p)
lr_acc_p = accuracy_score(y_test_p, lr_pred_p)
print("Priority Model")
print(f"Best Logistic Regression Parameters: {grid_search_lr_p.best_params_}")
print(f"Logistic Regression Test Accuracy (tuned): {lr_acc_p:.3f}")


# Model 2: Random Forest with GridSearchCV
print("\n🌲 Random Forest with GridSearchCV:")
# Define parameter grid for RandomForestClassifier - Expanded ranges
rf_param_grid = {
    "n_estimators": [100, 200, 300],  # Expanded n_estimators
    "max_depth": [15, 25, None],  # Expanded max_depth
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize GridSearchCV for Random Forest (Category)
grid_search_rf_c = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,  # Use all available cores
    verbose=1,
)
grid_search_rf_c.fit(X_train_c, y_train_c)

# Initialize GridSearchCV for Random Forest (Priority)
grid_search_rf_p = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,  # Use all available cores
    verbose=1,
)
grid_search_rf_p.fit(X_train_p, y_train_p)

# Get the best Random Forest model and its accuracy (Category)
rf_c = grid_search_rf_c.best_estimator_
rf_pred_c = rf_c.predict(X_test_c)
rf_acc_c = accuracy_score(y_test_c, rf_pred_c)
print("Category Model")
print(f"Best Random Forest Parameters: {grid_search_rf_c.best_params_}")
print(f"Random Forest Test Accuracy (tuned): {rf_acc_c:.3f}")

# Get the best Random Forest model and its accuracy (Priority)
rf_p = grid_search_rf_p.best_estimator_
rf_pred_p = rf_p.predict(X_test_p)
rf_acc_p = accuracy_score(y_test_p, rf_pred_p)
print("Priority Model")
print(f"Best Random Forest Parameters: {grid_search_rf_p.best_params_}")
print(f"Random Forest Test Accuracy (tuned): {rf_acc_p:.3f}")


# Choose best model (Category)
best_model_c = rf_c if rf_acc_c > lr_acc_c else lr_c
best_acc_c = max(lr_acc_c, rf_acc_c)
print(
    f"\n🏆 Best category model: {'Random Forest' if rf_acc_c > lr_acc_c else 'Logistic Regression'}"
)
print(f"Best accuracy for category model: {best_acc_c:.3f}")

# Choose best model (Priority)
best_model_p = rf_p if rf_acc_p > lr_acc_p else lr_p
best_acc_p = max(lr_acc_p, rf_acc_p)
print(
    f"\n🏆 Best priority model: {'Random Forest' if rf_acc_p > lr_acc_p else 'Logistic Regression'}"
)
print(f"Best priority accuracy: {best_acc_p:.3f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n📈 CREATING VISUALIZATIONS")
print("-" * 30)

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # 1. Category distribution
# df["category"].value_counts().plot(kind="bar", ax=axes[0, 0])
# axes[0, 0].set_title("Task Category Distribution")
# axes[0, 0].set_xlabel("Category")  # Added x-label
# axes[0, 0].set_ylabel("Count")  # Added y-label
# axes[0, 0].tick_params(axis="x", rotation=45)

# # 2. Priority distribution
# df["priority"].value_counts().plot(kind="pie", ax=axes[0, 1], autopct="%1.1f%%")
# axes[0, 1].set_title("Priority Distribution")

# # 3. Model comparison
# models = ["Logistic Regression (Tuned)", "Random Forest (Tuned)"]  # Updated labels
# accuracies = [lr_acc, rf_acc]
# axes[1, 0].bar(models, accuracies, color=["skyblue", "lightgreen"])
# axes[1, 0].set_title("Model Accuracy (Tuned)")  # Updated title
# axes[1, 0].set_ylabel("Accuracy")
# axes[1, 0].set_ylim(0, 1)  # Set y-limit for better visualization
# for i, acc in enumerate(accuracies):
#     axes[1, 0].text(i, acc + 0.01, f"{acc:.3f}", ha="center")

# # 4. Confusion matrix
# cm = confusion_matrix(y_test_c, rf_pred if rf_acc > lr_acc else lr_pred)
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=category_le.classes_,
#     yticklabels=category_le.classes_,
#     ax=axes[1, 1],
# )
# axes[1, 1].set_title("Confusion Matrix")
# axes[1, 1].set_xlabel("Predicted")
# axes[1, 1].set_ylabel("Actual")

# plt.tight_layout()
# plt.savefig("images/results.png", dpi=300, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# 1. Category distribution
df["category"].value_counts().plot(kind="bar", ax=axes[0, 0])
axes[0, 0].set_title("Task Category Distribution")
axes[0, 0].set_xlabel("Category")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Priority distribution
df["priority"].value_counts().plot(kind="pie", ax=axes[0, 1], autopct="%1.1f%%")
axes[0, 1].set_title("Priority Distribution")

# 3. Model comparison - Category
models_cat = ["Logistic Regression", "Random Forest"]
accuracies_cat = [lr_acc_c, rf_acc_c]
axes[1, 0].bar(models_cat, accuracies_cat, color=["skyblue", "lightgreen"])
axes[1, 0].set_title("Category Model Accuracy")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_ylim(0, 1)
for i, acc in enumerate(accuracies_cat):
    axes[1, 0].text(i, acc + 0.01, f"{acc:.3f}", ha="center")

# 4. Confusion matrix - Category
cm_cat = confusion_matrix(y_test_c, rf_pred_c if rf_acc_c > lr_acc_c else lr_pred_c)
sns.heatmap(
    cm_cat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=category_le.classes_,
    yticklabels=category_le.classes_,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Category Model Confusion Matrix")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")

# 5. Model comparison - Priority
models_pri = ["Logistic Regression", "Random Forest"]
accuracies_pri = [lr_acc_p, rf_acc_p]
axes[2, 0].bar(models_pri, accuracies_pri, color=["lightcoral", "lightseagreen"])
axes[2, 0].set_title("Priority Model Accuracy")
axes[2, 0].set_ylabel("Accuracy")
axes[2, 0].set_ylim(0, 1)
for i, acc in enumerate(accuracies_pri):
    axes[2, 0].text(i, acc + 0.01, f"{acc:.3f}", ha="center")

# 6. Confusion matrix - Priority
cm_pri = confusion_matrix(y_test_p, rf_pred_p if rf_acc_p > lr_acc_p else lr_pred_p)
sns.heatmap(
    cm_pri,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=priority_le.classes_,
    yticklabels=priority_le.classes_,
    ax=axes[2, 1],
)
axes[2, 1].set_title("Priority Model Confusion Matrix")
axes[2, 1].set_xlabel("Predicted")
axes[2, 1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("images/results.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================================================================
# TASK PREDICTION FUNCTIONS
# ==============================================================================


def predict_task_category(title, description=""):
    # Predict category for new task
    text = clean_and_lemmatize(
        title + " " + description
    )  # Use the new cleaning function
    vector = tfidf.transform([text])
    prediction = best_model_c.predict(vector)[0]
    probability = best_model_c.predict_proba(vector)[0]

    category = category_le.inverse_transform([prediction])[0]
    confidence = probability[prediction]

    return category, confidence


def predict_task_priority(title, description=""):
    # Predict priority for new task
    text = clean_and_lemmatize(
        title + " " + description
    )  # Use the new cleaning function
    vector = tfidf.transform([text])
    prediction = best_model_p.predict(vector)[0]
    probability = best_model_p.predict_proba(vector)[0]

    priority = priority_le.inverse_transform([prediction])[0]
    confidence = probability[prediction]

    return priority, confidence


# Test predictions
print("\n🎯 PREDICTION EXAMPLES:")
print("-" * 30)

test_tasks = [
    ("Complete Python assignment", "Write web scraper"),
    ("Doctor appointment", "Health checkup"),
    ("Team meeting", "Weekly standup"),
    ("Gym workout", "Chest training"),
    ("Read research paper", "ML optimization"),
]

for title, desc in test_tasks:
    category, confidence = predict_task_category(title, desc)
    print(f"Category: '{title}' → {category} ({confidence:.3f})")

    priority, confidence = predict_task_priority(title, desc)
    print(f"Priority: '{title}' → {priority} ({confidence:.3f})")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n📋 SUMMARY")
print("=" * 50)

print(f"""
✅ COMPLETED:
• NLP text preprocessing & tokenization
• TF-IDF vectorization ({X.shape[1]} features)
• ML models to categorize tasks (LR: {lr_acc_c:.3f}, RF: {rf_acc_c:.3f})
• ML models to prioritize tasks (LR: {lr_acc_p:.3f}, RF: {rf_acc_p:.3f})

📊 DATASET STATS:
• {len(df)} tasks across {df["category"].nunique()} categories
• Most common: {df["category"].value_counts().index[0]} ({df["category"].value_counts().iloc[0]} tasks)
• Best model accuracy for category model: {best_acc_c:.3f}
• Best model accuracy for priority model: {best_acc_p:.3f}
""")

# ==============================================================================
# SAVE THE MODELS, VECTORIZER, AND LABEL ENCODER
# ==============================================================================

joblib.dump(best_model_c, "models/categorize_model.pkl")
joblib.dump(best_model_p, "models/prioritize_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(category_le, "models/label_encoder.pkl")
