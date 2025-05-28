import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Student Performance",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2E75B6 0%, #1E3E74 100%) !important;
        }
        .main-title {
            font-size: 2.8rem !important;
            color: #2E75B6;
            text-align: center;
            padding: 20px 0;
        }
        .section-header {
            font-size: 1.6rem;
            color: #1E3E74;
            border-bottom: 3px solid #2E75B6;
            padding-bottom: 6px;
            margin-top: 20px;
        }
        .metric-card {
            background: #f0f2f6;
            border-radius: 12px;
            padding: 16px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-container {
            background: #ffffff;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("student_habits_performance.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure 'student_habits_performance.csv' exists.")
        st.stop()

@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure both 'best_model.pkl' and 'scaler.pkl' exist.")
        st.stop()

# Main app
df = load_data()
model, scaler = load_models()

# Sidebar filters
st.sidebar.header("🔍 Filter Students")
age_range = st.sidebar.slider("Select Age Range", int(df.age.min()), int(df.age.max()), (18, 22))
attendance_min = st.sidebar.slider("Minimum Attendance (%)", 0, 100, 70)
health_filter = st.sidebar.multiselect("Mental Health Rating", sorted(df.mental_health_rating.unique().tolist()), default=sorted(df.mental_health_rating.unique().tolist()))

filtered_df = df[
    (df.age.between(*age_range)) &
    (df.attendance_percentage >= attendance_min) &
    (df.mental_health_rating.isin(health_filter))
]

# Main title
st.markdown('<h1 class="main-title">Student Academic Performance Analyzer</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Data Analysis", "🤖 Model Insights", "🎯 Predictor"])

with tab1:
    st.markdown("""
    ## About This Dashboard

    This interactive analytics platform examines the relationship between student lifestyle factors and academic performance.
    Leveraging machine learning and data visualization, it provides insights into key performance drivers.

    **Key Features:**
    - Comprehensive dataset exploration
    - Interactive visualizations
    - Machine learning model evaluation
    - Performance prediction simulator
    """)

    st.image("https://images.unsplash.com/photo-1580582932707-520aed937b7b?auto=format&fit=crop&w=1920&q=80",
             use_container_width=True, caption="Education Analytics Concept")

    with st.expander("📁 Dataset Summary"):
        try:
            styled_df = filtered_df.describe().T.style.background_gradient(cmap="Blues")
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.error("Failed to render styled dataframe. Showing raw summary instead.")
            st.dataframe(filtered_df.describe().T, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(filtered_df, x="study_hours_per_day", nbins=30,
                          color_discrete_sequence=['#2E75B6'],
                          title="Daily Study Hours Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(filtered_df, y="exam_score", color_discrete_sequence=['#1E3E74'],
                    title="Exam Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Feature Relationships</div>', unsafe_allow_html=True)
    fig3 = px.scatter_matrix(filtered_df.select_dtypes(include=['number']),
                           dimensions=["study_hours_per_day", "sleep_hours", "attendance_percentage", "exam_score"],
                           color="exam_score", height=800)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    features = ["study_hours_per_day", "social_media_hours", "netflix_hours", "attendance_percentage", "sleep_hours"]
    X = filtered_df[features]
    y = filtered_df["exam_score"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <p style="font-size: 2.5rem; color: #1E3E74; margin: 10px 0;">
                {r2_score(y, y_pred):.2f} R² Score
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction Error</h3>
            <p style="font-size: 2.5rem; color: #1E3E74; margin: 10px 0;">
                {np.sqrt(mean_squared_error(y, y_pred)):.2f} RMSE
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Prediction Analysis</div>', unsafe_allow_html=True)
    fig4 = px.scatter(x=y, y=y_pred, trendline="ols",
                      labels={'x': 'Actual Scores', 'y': 'Predicted Scores'},
                      title="Actual vs Predicted Exam Scores")
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Performance Predictor</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        Adjust the lifestyle parameters to simulate student habits and predict academic performance.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            study_hours = st.slider("📚 Study Hours/Day", 0.0, 12.0, 3.0, 0.5)
            social_hours = st.slider("💬 Social Media Hours", 0.0, 10.0, 2.0, 0.5)
        with col2:
            netflix_hours = st.slider("🎬 Streaming Hours", 0.0, 10.0, 1.0, 0.5)
            attendance = st.slider("✅ Attendance (%)", 0, 100, 75)
        with col3:
            sleep = st.slider("😴 Sleep Hours/Night", 0.0, 12.0, 6.0, 0.5)

        input_df = pd.DataFrame([{
            "study_hours_per_day": study_hours,
            "social_media_hours": social_hours,
            "netflix_hours": netflix_hours,
            "attendance_percentage": attendance,
            "sleep_hours": sleep
        }])

        if st.button("🚀 Calculate Predicted Score", use_container_width=True):
            input_scaled = scaler.transform(input_df)
            predicted_score = model.predict(input_scaled)[0]
            st.markdown(f"""
            <div class="prediction-container">
                <h3 style="color: #1E3E74; margin: 0;">Predicted Exam Score</h3>
                <div style="font-size: 3.5rem; color: #2E75B6; text-align: center; margin: 20px 0;">
                    {predicted_score:.1f}/100
                </div>
                <p style="text-align: center; color: #666;">
                    Based on current lifestyle parameters
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 20px;">
    Educational Analytics Dashboard • Streamlit • Data Science Project
</div>
""", unsafe_allow_html=True)
