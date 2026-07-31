# ==========================================
# MovieIQ - Movie Success Prediction Dashboard
# ==========================================

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
import re

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🎬 MovieIQ Dashboard")
st.markdown("### Movie Success Prediction using Machine Learning")

st.markdown("---")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/movies.csv")

df = load_data()

import numpy as np

df = load_data()

# Create target column
df["success"] = np.where(
    df["revenue"] > df["budget"],
    1,
    0
)

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")

model = load_model()



st.sidebar.title("🎯 Filters")

# Extract genre names from JSON-like strings
def extract_genres(genre_string):
    try:
        genre_list = ast.literal_eval(genre_string)
        return [g["name"] for g in genre_list]
    except:
        return []

df["genre_list"] = df["genres"].apply(extract_genres)

all_genres = sorted(
    set(
        genre
        for genres in df["genre_list"]
        for genre in genres
    )
)

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + all_genres
)

min_vote = st.sidebar.slider(
    "Minimum Vote Average",
    0.0,
    10.0,
    5.0
)

filtered_df = df.copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["genre_list"].apply(
            lambda x: selected_genre in x
        )
    ] 

filtered_df = filtered_df[
    filtered_df["vote_average"] >= min_vote
]

# ----------------------------
# KPI Calculations
# ----------------------------
total_movies = len(filtered_df)

successful_movies = filtered_df["success"].sum()

success_rate = (successful_movies / total_movies) * 100

avg_budget = filtered_df["budget"].mean()

avg_revenue = filtered_df["revenue"].mean()

st.subheader("📊 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎬 Total Movies",
        f"{total_movies:,}"
    )

with col2:
    st.metric(
        "✅ Successful Movies",
        int(successful_movies)
    )

with col3:
    st.metric(
        "📈 Success Rate",
        f"{success_rate:.1f}%"
    )

with col4:
    st.metric(
        "💰 Avg Revenue",
        f"${avg_revenue:,.0f}"
    )

st.markdown("---")

st.subheader("💰 Budget vs Revenue")

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    filtered_df["budget"],
    filtered_df["revenue"],
    alpha=0.6
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Budget (Log Scale)")
ax.set_ylabel("Revenue (Log Scale)")
ax.set_title("Budget vs Revenue")

st.pyplot(fig)

st.subheader("🎭 Top Movie Genres")
# Count genres from the cleaned genre_list
genre_count = (
    filtered_df["genre_list"]
    .explode()
    .value_counts()
)
fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
    genre_count.index[:10],
    genre_count.values[:10]
)

ax.set_xlabel("Genre")
ax.set_ylabel("Number of Movies")
ax.set_title("Top 10 Movie Genres")

plt.xticks(rotation=45)

st.pyplot(fig)


st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    filtered_df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

st.subheader("📈 Feature Importance")

importance = pd.DataFrame({
    "Feature": ["Budget", "Popularity", "Runtime", "Vote Average"],
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    importance["Feature"],
    importance["Importance"]
)

ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.set_title("Feature Importance")

st.pyplot(fig)

st.markdown("---")

st.subheader("📑 Statistical Analysis")

st.write("### T-Test")

st.info("""
Null Hypothesis (H₀):
There is no significant difference in popularity between successful and unsuccessful movies.

Result:
Since the p-value is less than 0.05, we reject the null hypothesis.
Popularity differs significantly between successful and unsuccessful movies.
""")

st.write("### Chi-Square Test")

st.info("""
Null Hypothesis (H₀):
Movie genre and success are independent.

Result:
Since the p-value is less than 0.05, we reject the null hypothesis.
Genre and movie success are significantly associated.
""")

st.markdown("---")

st.header("🤖 Predict Movie Success")

budget = st.number_input(
    "Budget",
    min_value=1000,
    value=5000000
)

popularity = st.number_input(
    "Popularity",
    value=50.0
)

runtime = st.number_input(
    "Runtime",
    value=120
)

vote_average = st.slider(
    "Vote Average",
    0.0,
    10.0,
    7.0
)

if st.button("Predict"):

    prediction = model.predict([[
        budget,
        popularity,
        runtime,
        vote_average
    ]])

    probability = model.predict_proba([[
        budget,
        popularity,
        runtime,
        vote_average
    ]])

    confidence = probability[0][prediction[0]] * 100

    if prediction[0] == 1:
        st.success("🎉 This movie is likely to be Successful!")
    else:
        st.error("❌ This movie is likely to be Unsuccessful.")

    st.write(f"### Confidence: {confidence:.2f}%")

    st.progress(int(confidence))

    st.markdown("---")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="filtered_movies.csv",
    mime="text/csv"
)


