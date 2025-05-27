import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load dataset
try:
    df = pd.read_csv("student_habits_performance.csv")
except FileNotFoundError:
    st.error("❌ File 'student_habits_performance.csv' not found.")
    st.stop()

# App title
st.title("📊 Student Academic Performance Dashboard")

# Overview tab content
st.header("📌 Project Overview")
st.markdown("""
This project analyzes how lifestyle habits and background factors affect academic performance using data science.

**Dataset Highlights:**
- 1000 students
- Features include study hours, sleep, mental health, parental education, and more
- Target: `exam_score`
""")

# Show the top 10 records
st.subheader("🔍 Preview of the Dataset")
st.dataframe(df.head(10), use_container_width=True)

# EDA Section
st.header("📈 Exploratory Data Analysis")

# Distribution of Study Hours
st.subheader("Study Hours per Day Distribution")
fig1 = px.histogram(df, x="study_hours_per_day", nbins=30, color_discrete_sequence=['#0A66C2'])
st.plotly_chart(fig1, use_container_width=True)

# Distribution of Exam Scores
st.subheader("Exam Score Distribution")
fig2 = px.histogram(df, x="exam_score", nbins=30, color_discrete_sequence=['#2E7D32'])
st.plotly_chart(fig2, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap (Numeric Features)")
corr = df.select_dtypes(include=['float64', 'int64']).corr()
fig3 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale="Viridis"))
st.plotly_chart(fig3, use_container_width=True)

# Model Evaluation Section
st.header("🤖 Model Evaluation")

# Load the model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load preprocessed data (only numeric for now)
features = ["study_hours_per_day", "social_media_hours", "netflix_hours", "attendance_percentage", "sleep_hours"]
X = df[features]
y = df["exam_score"]

# Scale the selected features
X_scaled = scaler.transform(X)

# Predict and evaluate
y_pred = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, y_pred))  # Manual square root
r2 = r2_score(y, y_pred)

# Display performance
st.subheader("📋 Model Performance")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R² Score", f"{r2:.2f}")

# Actual vs Predicted Plot
st.subheader("📉 Actual vs Predicted Exam Scores")
fig4 = px.scatter(
    x=y,
    y=y_pred,
    labels={'x': 'Actual Score', 'y': 'Predicted Score'},
    title="Actual vs Predicted Scores",
    trendline="ols"
)
st.plotly_chart(fig4, use_container_width=True)

st.header("🎯 Predict Exam Score")

st.markdown("Adjust the sliders below to simulate a student's lifestyle and predict their exam score.")

# Input sliders
study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 3.0)
social_hours = st.slider("Social Media Hours per Day", 0.0, 10.0, 2.0)
netflix_hours = st.slider("Netflix Hours per Day", 0.0, 10.0, 1.0)
attendance = st.slider("Attendance Percentage", 0, 100, 75)
sleep = st.slider("Sleep Hours per Night", 0.0, 12.0, 6.0)

# Format input
input_df = pd.DataFrame([{
    "study_hours_per_day": study_hours,
    "social_media_hours": social_hours,
    "netflix_hours": netflix_hours,
    "attendance_percentage": attendance,
    "sleep_hours": sleep
}])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
predicted_score = model.predict(input_scaled)[0]

# Display result
st.success(f"📘 Predicted Exam Score: **{predicted_score:.2f}**")

