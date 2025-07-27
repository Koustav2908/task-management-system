# 🧠 AI-Powered Task Management System

An intelligent task management system that uses machine learning and NLP to automatically categorize tasks and assign priority levels, making task handling smarter and more efficient. Built collaboratively by a team of four contributors, each responsible for essential modules of the system.

<div align="center"> <img src="images/results.png" alt="Model Results" width="600"/> </div>

---

## 🚀 Features

-   📂 Task Categorization (e.g., **work**, **health**, **personal**, etc.)

-   🔥 Priority Prediction (e.g., **high**, **medium**, **low**)

-   🧠 NLP-powered text preprocessing and `TF-IDF` vectorization

-   🤖 ML models: `Logistic Regression` & `Random Forest`

-   📊 Visual insights on model performance and predictions

-   📎 Interactive `Streamlit` dashboard

-   📤 Exportable `CSV` history

---

## 📦 Installation

1. Clone this repo

```bash
git clone https://github.com/Koustav2908/task-management-system.git
cd task-management-system
```

2. Install dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

---

3. Download SpaCy English model

```bash
python -m spacy download en_core_web_sm
```

## 🧪 Training the Model

Use `train.py` to train the task categorization and priority models.

```bash
python train.py
```

This will:

-   Preprocess and lemmatize text using SpaCy

-   Vectorize using TF-IDF

-   Train models with GridSearchCV

-   Evaluate performance

-   Save best models and visualizations in the `models/` and `images/` folders

---

## 💻 Running the Dashboard

```bash
streamlit run dashboard.py
```

Features:

-   Input task title and description

-   Predicts category and priority

-   Shows keyword weights

-   Logs predictions and allows CSV download

-   Sends Slack alert on high-priority detection (optional)

---

## 📈 Sample Output

---

| Title                   | Description                               | Category  | Priority |
| ----------------------- | ----------------------------------------- | --------- | -------- |
| Complete Python project | Finish making the dashboard and UI.       | Education | High     |
| Doctor appointment      | Book morning slot for your dentist.       | Health    | Medium   |
| Weekly gym workout      | Finish your weekly workout in 30 minutes. | Fitness   | Low      |

You can download your tasks too in `.csv` format.

---

## 🤝 Contributors

1. 📊 NLP Preprocessing, Categorization Model Training, & Visualization - [Harsha Vardhan Reddy](https://github.com/HarshaVardhan8a)
2. 🧠 Data Collection, & Prioritization Model Training - [Koustav Chatterjee](https://github.com/Koustav2908)
3. 🎨 Making the dashboard using Streamlit UI - [Dinesh S.J.](https://github.com/Dineshsj3002)
4. 📚 Data Collection, & Guidance - [Satya Ranjan Sahoo](https://github.com/Srs-satya)
