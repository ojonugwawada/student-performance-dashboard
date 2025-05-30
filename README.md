# 🎓 Student Academic Performance Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://student-insight.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/Repo-GitHub-blue?logo=github)](https://github.com/ojonugwawada/student-performance-dashboard)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)

An interactive analytics and prediction platform exploring how lifestyle habits, mental health, and academic behaviors impact student performance — built using data science and machine learning.

---

## 📌 Key Features

- 📊 Multi-tab interactive dashboard with rich visualizations
- 🧠 Machine learning model to estimate student exam scores
- 📈 Exploratory Data Analysis (EDA) with dynamic filters
- 🧪 Feature Engineering: correlation heatmap and distribution visualizer
- 🎯 Predictive Simulator: input study, sleep, media habits to forecast performance
- 🔄 Actual vs Predicted exam score comparisons
- 🎛️ Sidebar filters by age, attendance, and mental health rating

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:
  - `pandas`, `numpy`, `joblib`
  - `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`

---

## 🚀 Live App

Try the live app here:  
👉 [https://student-insight.streamlit.app](https://student-insight.streamlit.app)

---

## 🛠️ Installation Instructions

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ojonugwawada/student-performance-dashboard.git
   cd student-performance-dashboard
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 🧑‍💻 Usage Instructions

- Use the sidebar filters to explore student attributes (e.g., age group, attendance, sleep habits).
- Navigate between EDA and Prediction tabs.
- Enter values in the predictor tab to simulate exam score predictions.
- Compare actual scores against machine learning-predicted values.

---

## 🖼️ Screenshots

### 📋 Dashboard Overview  
![Overview](assets/overview-sample.png)

### 🎯 Predictive Score Output  
![Prediction](assets/prediction-example.png)

---

## 🗂️ Repository Contents

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `student_habits_performance.csv`: Dataset used for model training and dashboard
- `scaler.pkl`, `best_model.pkl`: Pre-trained model artifacts
- `assets/`: Folder containing visual assets for the README

---

## 🧽 Future Enhancements

- 📦 Add login functionality for user-specific tracking
- 📊 Expand dataset with more behavioral features
- 🔀 Incorporate real-time data ingestion for live academic dashboards
- 🧙️ Deploy as an API-integrated microservice

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributions & Feedback

Feel free to fork the project, raise issues, or submit pull requests to improve this tool.

📬 Let’s connect on [LinkedIn](https://www.linkedin.com/in/ojonugwa-wada-47ba55b7)

---

© 2025 Ojonugwa Wada | Academic & Educational Research
