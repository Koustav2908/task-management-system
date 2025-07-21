# Week 2: AI-Powered Task Management System - Simplified Version
# Sessions 4-6: NLP Basics, Text Vectorization, ML Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy # Added spacy import for lemmatization, though the simplified version's clean_text doesn't use it directly
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # Added GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("WEEK 2: AI-POWERED TASK MANAGEMENT SYSTEM")
print("="*50)

# Load dataset
df = pd.read_csv('tasks.csv')
print(f"Dataset loaded: {df.shape[0]} tasks, {df['category'].nunique()} categories")

# ==============================================================================
# SESSION 4: NLP BASICS - TOKENIZATION, POS, LEMMATIZATION
# ==============================================================================

print("\n📚 SESSION 4: NLP BASICS")
print("-" * 30)

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Fallback to simple cleaning if spaCy model is not available
    def clean_and_lemmatize(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    nlp = None # Set nlp to None if model not loaded

def clean_and_lemmatize(text):
    """
    Advanced text cleaning and lemmatization using spaCy.
    Removes stop words, punctuation, and converts words to their base form.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Replace non-alphabetic with space
    
    if nlp: # Use spaCy only if the model was loaded successfully
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.lemma_) > 2]
        return " ".join(tokens)
    else: # Fallback to simple cleaning if spaCy model is not available
        return ' '.join(text.split())


# Clean and combine text using the enhanced function
df['title_desc'] = (df['title'].fillna('') + " " + df['description'].fillna('')).apply(clean_and_lemmatize)

print("✅ Text preprocessing completed")
print(f"Sample cleaned text: {df['title_desc'].iloc[0]}")

# Simple tokenization example (using the new cleaned text)
sample_text = df['title_desc'].iloc[0]
tokens = sample_text.split()
print(f"Tokens: {tokens[:5]}...")

# ==============================================================================
# SESSION 5: TEXT VECTORIZATION - TF-IDF
# ==============================================================================

print("\n📊 SESSION 5: TEXT VECTORIZATION")
print("-" * 30)

# TF-IDF Vectorization with increased max_features
tfidf = TfidfVectorizer(
    max_features=1500, # Increased from 500
    stop_words='english', # This might be redundant if spaCy already removed them, but good as a safeguard
    ngram_range=(1, 2),
    min_df=2
)

# Transform text to vectors
X = tfidf.fit_transform(df['title_desc']) # Use the new 'title_desc' column
print(f"✅ TF-IDF matrix created: {X.shape}")

# Show top features by category
print("\n🔍 Top terms by category:")
for category in df['category'].unique()[:3]:  # Show only 3 categories
    category_data = df[df['category'] == category]['title_desc'] # Use the new 'title_desc' column
    category_tfidf = tfidf.transform(category_data)
    mean_scores = np.array(category_tfidf.mean(axis=0)).flatten()
    top_words = [tfidf.get_feature_names_out()[i] for i in mean_scores.argsort()[-3:][::-1]]
    print(f"{category}: {top_words}")

# ==============================================================================
# SESSION 6: ML MODELS FOR TASK CATEGORIZATION
# ==============================================================================

print("\n🤖 SESSION 6: ML MODELS")
print("-" * 30)

# Prepare data
y = df['category']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")

# Model 1: Logistic Regression with GridSearchCV
print("\n📊 Logistic Regression with GridSearchCV:")
# Define parameter grid for LogisticRegression - Expanded C range
lr_param_grid = {
    'C': [0.1, 1, 10, 100], # Expanded Regularization strength
    'solver': ['liblinear', 'lbfgs'], # Different solvers
    'max_iter': [1000, 2000, 3000] # Expanded max_iter
}

# Initialize GridSearchCV for Logistic Regression
grid_search_lr = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=5, # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)
grid_search_lr.fit(X_train, y_train)

# Get the best Logistic Regression model and its accuracy
lr = grid_search_lr.best_estimator_
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Best Logistic Regression Parameters: {grid_search_lr.best_params_}")
print(f"Logistic Regression Test Accuracy (tuned): {lr_acc:.3f}")


# Model 2: Random Forest with GridSearchCV
print("\n🌲 Random Forest with GridSearchCV:")
# Define parameter grid for RandomForestClassifier - Expanded ranges
rf_param_grid = {
    'n_estimators': [100, 200, 300], # Expanded n_estimators
    'max_depth': [15, 25, None], # Expanded max_depth
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5, # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)
grid_search_rf.fit(X_train, y_train)

# Get the best Random Forest model and its accuracy
rf = grid_search_rf.best_estimator_
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Best Random Forest Parameters: {grid_search_rf.best_params_}")
print(f"Random Forest Test Accuracy (tuned): {rf_acc:.3f}")


# Choose best model
best_model = rf if rf_acc > lr_acc else lr
best_acc = max(lr_acc, rf_acc)
print(f"\n🏆 Best model: {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'}")
print(f"Best accuracy: {best_acc:.3f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n📈 CREATING VISUALIZATIONS")
print("-" * 30)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Category distribution
df['category'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Task Category Distribution')
axes[0,0].set_xlabel('Category') # Added x-label
axes[0,0].set_ylabel('Count') # Added y-label
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Priority distribution
df['priority'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
axes[0,1].set_title('Priority Distribution')

# 3. Model comparison
models = ['Logistic Regression (Tuned)', 'Random Forest (Tuned)'] # Updated labels
accuracies = [lr_acc, rf_acc]
axes[1,0].bar(models, accuracies, color=['skyblue', 'lightgreen'])
axes[1,0].set_title('Model Accuracy (Tuned)') # Updated title
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_ylim(0, 1) # Set y-limit for better visualization
for i, acc in enumerate(accuracies):
    axes[1,0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')

# 4. Confusion matrix
cm = confusion_matrix(y_test, rf_pred if rf_acc > lr_acc else lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1,1])
axes[1,1].set_title('Confusion Matrix')
axes[1,1].set_xlabel("Predicted") # Added x-label
axes[1,1].set_ylabel("Actual") # Added y-label

plt.tight_layout()
plt.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# TASK PREDICTION FUNCTION
# ==============================================================================

def predict_task(title, description=""):
    # Predict category for new task
    text = clean_and_lemmatize(title + ' ' + description) # Use the new cleaning function
    vector = tfidf.transform([text])
    prediction = best_model.predict(vector)[0]
    probability = best_model.predict_proba(vector)[0]

    category = le.inverse_transform([prediction])[0]
    confidence = probability[prediction]

    return category, confidence

# Test predictions
print("\n🎯 PREDICTION EXAMPLES:")
print("-" * 30)

test_tasks = [
    ("Complete Python assignment", "Write web scraper"),
    ("Doctor appointment", "Health checkup"),
    ("Team meeting", "Weekly standup"),
    ("Gym workout", "Chest training"),
    ("Read research paper", "ML optimization")
]

for title, desc in test_tasks:
    category, confidence = predict_task(title, desc)
    print(f"'{title}' → {category} ({confidence:.3f})")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n📋 WEEK 2 SUMMARY")
print("=" * 50)

print(f"""
✅ COMPLETED:
• Session 4: NLP text preprocessing & tokenization
• Session 5: TF-IDF vectorization ({X.shape[1]} features)
• Session 6: ML models (LR: {lr_acc:.3f}, RF: {rf_acc:.3f})

📊 DATASET STATS:
• {len(df)} tasks across {df['category'].nunique()} categories
• Most common: {df['category'].value_counts().index[0]} ({df['category'].value_counts().iloc[0]} tasks)
• Best model accuracy: {best_acc:.3f}

🚀 READY FOR WEEK 3: Time-series forecasting & smart predictions
""")

print("="*50)
print("Week 2 completed successfully! 🎉")
print("="*50)