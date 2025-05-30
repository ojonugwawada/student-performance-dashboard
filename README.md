# 🎓 Student Academic Performance Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://student-insight.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/Repo-GitHub-blue?logo=github)](https://github.com/ojonugwawada/student-performance-dashboard)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)

An interactive analytics and prediction platform exploring how lifestyle habits, mental health, and academic behaviors impact student performance — powered by data science and machine learning.

---

## 📌 Key Features

- 📊 Multi-tab dashboard with interactive visualizations
- 🧠 ML-powered predictor for estimating student exam scores
- 🔍 EDA tools with histogram, box plot, and scatter matrix
- 🧪 Feature Engineering tab: correlation heatmap, distribution plots
- 🎛️ Filters: age range, attendance threshold, mental health rating
- 🌗 Theme toggle: Light/Dark display modes
- 📅 Prediction report export to CSV
- 📖 User guide tab for onboarding new users

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:
  - `pandas`, `numpy`, `joblib`
  - `scikit-learn`, `matplotlib`, `seaborn`
  - `plotly`, `statsmodels`

---

## 🚀 Live App

👉 [Launch Live Dashboard](https://student-insight.streamlit.app)

---

## 🛠️ Installation Instructions

To run this project locally:

```bash
# Clone repository
git clone https://github.com/ojonugwawada/student-performance-dashboard.git
cd student-performance-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

---

## 🧑‍💻 How to Use

- Use the **sidebar filters** to refine student data.
- Navigate between **tabs** to explore features:
  - **Overview** – Project intro and summary stats
  - **Data Analysis** – Histogram, boxplot, scatter matrix
  - **Feature Engineering** – Heatmaps and distribution explorer
  - **Model Insights** – R² and RMSE metrics + actual vs predicted
  - **Predictor** – Simulate exam scores based on lifestyle habits
  - **User Guide** – Instructions and dashboard walkthrough
- Use **dark mode toggle** for preferred theme.
- Export prediction results with the **CSV download button**.

---

## 🖼️ Screenshots

### 📋 Dashboard Overview  
![Overview](assets/overview-sample.png)

### 🎯 Predicted Score Output  
![Prediction](assets/prediction-example.png)

---

## 🗂️ Repository Contents

- `app.py` — Main Streamlit app
- `requirements.txt` — Project dependencies
- `student_habits_performance.csv` — Dataset used
- `scaler.pkl`, `best_model.pkl` — Pre-trained model files
- `assets/` — Visual assets for README

---

## 🧽 Future Enhancements

- 🧑‍🎓 User login for personalized tracking
- 📊 Integration with larger real-world datasets
- 🛡️ Live data streaming for real-time predictions
- ⚙️ Deploy as an API/microservice backend

---

## 📜 License

Licensed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributions & Feedback

Feel free to fork, raise issues, or submit pull requests!

📬 Let’s connect on [LinkedIn](https://www.linkedin.com/in/ojonugwa-wada-47ba55b7)

---

© 2025 Ojonugwa Wada | Academic Research & Analytics