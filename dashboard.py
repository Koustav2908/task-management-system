import datetime
import os
from io import StringIO

import joblib
import pandas as pd
import requests
import streamlit as st


# Utility to load models
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Paths
MODEL_DIR = "models"
priority_model = load_model(os.path.join(MODEL_DIR, "priority_model.pkl"))
category_model = load_model(os.path.join(MODEL_DIR, "category_model.pkl"))
vectorizer = load_model(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
priority_le = load_model(os.path.join(MODEL_DIR, "priority_label_encoder.pkl"))
category_le = load_model(os.path.join(MODEL_DIR, "category_label_encoder.pkl"))

# Page config with light theme hint
st.set_page_config(page_title="Insightful Task Predictor", layout="centered")

# Custom CSS for light theme
st.markdown(
    """
<style>
    .reportview-container, .sidebar .sidebar-content {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("Predict task Category & Priority. Logs history, and export CSV.")

# Main title
st.title("🔮 Insightful Task Category & Priority Predictor")
st.write(
    "Enter a task title and description to see predictions, insights, and history log."
)

# User inputs
title = st.text_input("📌 Task Title")
desc = st.text_area("📝 Task Description")

# Predict
if st.button("Predict"):
    if not title or not desc:
        st.warning("Please fill in both fields!")
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{title} {desc}"
        vec = vectorizer.transform([text])

        # Predict
        prio_pred = priority_model.predict(vec)
        cat_pred = category_model.predict(vec)
        prio_label = priority_le.inverse_transform(prio_pred)[0]
        cat_label = category_le.inverse_transform(cat_pred)[0]

        # Display
        st.header("Results")
        st.success(f"🔥 Priority: {prio_label}")
        st.success(f"📂 Category: {cat_label}")

        # Probabilities
        try:
            prio_proba = priority_model.predict_proba(vec)[0]
            cat_proba = category_model.predict_proba(vec)[0]
            st.subheader("Prediction Probabilities")
            st.bar_chart({c: prio_proba[i] for i, c in enumerate(priority_le.classes_)})
            st.bar_chart({c: cat_proba[i] for i, c in enumerate(category_le.classes_)})
        except Exception:
            pass

        # Top Keywords
        st.subheader("Top Keywords by TF-IDF")
        feature_names = vectorizer.get_feature_names_out()
        sorted_indices = vec.toarray()[0].argsort()[::-1][:10]
        keywords = feature_names[sorted_indices]
        st.write(", ".join(keywords))

        # Add to history
        record = {
            "Timestamp": timestamp,
            "Title": title,
            "Description": desc,
            "Priority": prio_label,
            "Category": cat_label,
        }
        st.session_state.history.append(record)

        # Slack alert on high priority
        webhook = os.getenv("SLACK_WEBHOOK_URL")
        if prio_label.lower() == "high" and webhook:
            try:
                msg = {
                    "text": f"*High Priority Task*\nTitle: {title}\nCategory: {cat_label}\nTime: {timestamp}"
                }
                requests.post(webhook, json=msg, timeout=3)
                st.info("Slack alert sent for high priority.")
            except Exception:
                st.error("Failed to send Slack alert.")

# History panel
if st.session_state.history:
    st.subheader("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    # Export to CSV
    csv_buffer = StringIO()
    hist_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv_buffer.getvalue(),
        file_name="prediction_history.csv",
    )
