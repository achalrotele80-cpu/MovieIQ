# 🎬 MovieIQ - Movie Success Prediction Dashboard

## 📌 Project Overview

MovieIQ is an interactive Streamlit dashboard that analyzes movie performance data and predicts whether a movie is likely to be successful using Machine Learning.

In this project, a movie is classified as successful when:

**Revenue > Budget**

The project combines Exploratory Data Analysis (EDA), Statistical Hypothesis Testing, Machine Learning Classification, and an interactive Streamlit application to understand the factors influencing movie success.

---

# 🎯 Business Problem

Predicting movie success is valuable because movie production involves significant financial investment and uncertainty.

### Stakeholders:

**1. Movie Studios / Production Houses**

* Can estimate the potential success of a movie before production decisions.
* Helps in budget planning and risk reduction.

**2. Investors / Producers**

* Can use predictions to evaluate investment opportunities and expected returns.

---

# 🎯 Project Objectives

The main objectives of MovieIQ are:

* Analyze historical movie performance data.
* Identify factors associated with movie success.
* Perform statistical analysis to find meaningful relationships.
* Build a machine learning model to predict success.
* Create an interactive dashboard for business users.

---

# 📂 Dataset

The dataset contains movie information including:

| Feature      | Description             |
| ------------ | ----------------------- |
| budget       | Movie production budget |
| revenue      | Movie earnings          |
| popularity   | Popularity score        |
| runtime      | Movie duration          |
| vote_average | Audience rating         |
| title        | Movie title             |
| genres       | Movie categories        |

Target Variable:

```
success = 1  → Revenue > Budget
success = 0  → Revenue <= Budget
```

---

# 🛠 Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Statistical Analysis

* SciPy

### Machine Learning

* Scikit-Learn
* Random Forest Classifier

### Deployment

* Streamlit

### Model Saving

* Joblib

---

# 📊 Exploratory Data Analysis

The project explores:

### Budget vs Revenue

Analysis of whether movies with higher budgets generally generate higher revenue.

### Genre Analysis

Identifies:

* Most common movie genres
* Genres associated with higher success rates

### Feature Analysis

Examines relationships between:

* Popularity
* Runtime
* Vote Average
* Movie Success

### Correlation Analysis

A correlation heatmap was generated to identify relationships between numerical variables.

---

# 📈 Statistical Testing

## T-Test

### Hypothesis:

**Null Hypothesis (H₀):**
There is no significant difference in popularity between successful and unsuccessful movies.

**Alternative Hypothesis (H₁):**
Successful and unsuccessful movies have significantly different popularity values.

Significance level:

```
α = 0.05
```

---

## Chi-Square Test

### Hypothesis:

**Null Hypothesis (H₀):**
Movie genre and success are independent.

**Alternative Hypothesis (H₁):**
Movie genre and success are associated.

---

# 🤖 Machine Learning Model

## Algorithm

Random Forest Classifier

Random Forest combines multiple decision trees and uses their combined predictions to improve accuracy and reduce overfitting.

---

## Features Used

Input Features:

```
budget
popularity
runtime
vote_average
```

Target:

```
success
```

Excluded:

* Movie title (text feature)
* Revenue (directly determines success and causes data leakage)

---

# 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

The trained model was saved as:

```
models/random_forest.pkl
```

The Streamlit application loads this trained model to provide real-time predictions.

---

# 🚀 Streamlit Dashboard Features

The dashboard includes:

✔ Executive KPI Dashboard

✔ Genre Filtering

✔ Minimum Vote Average Filtering

✔ Budget vs Revenue Visualization

✔ Genre Analysis

✔ Correlation Heatmap

✔ Feature Importance

✔ Statistical Test Results

✔ Movie Success Prediction

✔ Download Filtered Dataset

---

# 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run MovieIQ.py
```

---

# 📁 Project Structure

```
MovieIQ/
│
├── data/
│   └── movies.csv
│
├── models/
│   └── random_forest.pkl
│
├── notebook/
│   └── MovieIQ_EDA.ipynb
│
├── MovieIQ.py
├── requirements.txt
├── README.md
```

---

# 🌐 Deployment

Live Streamlit Application:

(Add deployment link here)

---

# 🔮 Future Improvements

* Add more movie features such as actors, directors, production companies.
* Experiment with advanced models like XGBoost and Gradient Boosting.
* Add real-time movie data API integration.
* Improve prediction accuracy with additional feature engineering.

---

# 👨‍💻 Author

**Achal Rotele**

Data Analytics
