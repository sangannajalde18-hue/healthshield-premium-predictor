# 🛡️ HealthShield Premium Predictor

### *Predicting healthcare premiums through intelligent risk assessment.*

HealthShield Premium Predictor is an end-to-end Machine Learning web application designed to estimate a user's annual health insurance premium based on their medical history, lifestyle patterns, and demographic attributes.

This project leverages **Gradient Boosting Regression** for premium prediction and integrates a **Flask-based deployment pipeline** with an interactive frontend interface for real-time predictions.

---

## ✨ Project Overview

Insurance pricing often depends on multiple hidden variables such as:

- Age  
- BMI  
- Medical expenditure  
- Smoking habits  
- Alcohol consumption  
- Pre-existing conditions  
- Hospitalization history  
- Major medical procedures  

HealthShield simplifies this process by allowing users to input their health-related attributes and instantly receive an estimated annual insurance premium.

---

## 🎯 Problem Statement

Traditional insurance premium estimation processes are often:

- Time-consuming  
- Non-transparent  
- Dependent on manual underwriting  
- Difficult for customers to understand  

This project aims to create a **data-driven premium estimation system** that improves transparency and provides users with quick insurance cost predictions.

---

# 🧠 Machine Learning Pipeline

### Data Preprocessing
- Handling categorical and numerical features
- One Hot Encoding
- Feature transformation using `ColumnTransformer`
- Data splitting using train-test methodology

---

### Models Trained
The following regression models were evaluated:

| Model | R² Score |
|--------|------------|
| Random Forest Regressor | 0.924 |
| Gradient Boosting Regressor | **0.930** ✅ |
| XGBoost Regressor | 0.892 |

Gradient Boosting Regressor delivered the best performance and was selected for deployment.

---

# ⚙️ Tech Stack

### Machine Learning
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Pickle  

### Backend
- Flask  

### Frontend
- HTML  
- CSS  

### Development Tools
- VS Code  
- Git  
- GitHub  

---

# 📥 Input Features

Users provide:

- Age  
- BMI  
- Annual Medical Cost  
- Hospitalization History  
- Hypertension Status  
- Diabetes Status  
- Arthritis Status  
- Mental Health Conditions  
- Major Procedure History  
- Smoking Frequency  
- Alcohol Consumption Frequency  
- Gender  

---

# 📤 Output

✅ Predicted Annual Health Insurance Premium

---

# 🖥️ Application Workflow

```bash
User Input → Data Preprocessing → Trained ML Model → Premium Prediction → Web Interface Output
```

---

# 📂 Project Structure

```bash
HealthShield-Premium-Predictor/
│
├── app.py
├── gboost_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

---

# 🚀 Installation & Setup

### Clone Repository

```bash
git clone https://github.com/your-username/HealthShield-Premium-Predictor.git
```

---

### Navigate to Project Directory

```bash
cd HealthShield-Premium-Predictor
```

---

### Create Virtual Environment

```bash
python -m venv myenv
```

---

### Activate Environment

```bash
myenv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Application

```bash
python app.py
```

---

### Open Browser

```bash
http://127.0.0.1:5000/
```

---

# 📸 Application Preview

_Add screenshots of your UI here for better portfolio presentation._

Example:

- Homepage UI  
- Input Form  
- Prediction Output Screen  

---

# 🔍 Key Highlights

✔ End-to-End ML Deployment Project  
✔ Real-time Prediction System  
✔ Flask Web Integration  
✔ Model Serialization using Pickle  
✔ User-Friendly Interface  
✔ Comparative Model Evaluation  

---

# 🌱 Future Enhancements

- React Frontend Integration  
- Cloud Deployment (AWS / Render / Azure)  
- BMI auto-calculation using height & weight  
- Insurance recommendation engine  
- User authentication system  
- Explainable AI integration  

---

# 📌 Business Use Cases

This system can be useful for:

- Health Insurance Companies  
- InsurTech Startups  
- Risk Assessment Teams  
- Healthcare Analytics Platforms  

---

# 👩‍💻 Author

**Sanganna Jalde**  
Aspiring Data Scientist | Machine Learning Enthusiast 

---

## ⭐ If you found this project interesting, consider giving it a star.
