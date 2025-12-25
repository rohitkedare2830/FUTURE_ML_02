# 🎯 Attrition Prediction Engine  
**Production-ready Machine Learning application for employee retention analysis**

---

## 🔍 What This Project Demonstrates

This project showcases my ability to **design, train, and deploy an end-to-end machine learning system** that solves a real business problem: **employee attrition**.

It highlights skills in:
- Machine learning model development  
- Feature engineering & preprocessing  
- Business-focused evaluation metrics  
- Deployment using Streamlit  
- Clean UI/UX for decision-makers  

---

## 🧠 Problem Statement

Employee attrition increases:
- Hiring and onboarding costs  
- Loss of experienced talent  
- Operational instability  

The goal of this system is to **predict churn risk early** so HR teams can take **proactive retention actions** instead of reactive decisions.

---

## ⚙️ Solution Overview

- Built a **binary classification model** to predict employee churn  
- Optimized the model for **high recall** to reduce false negatives  
- Applied a **custom probability threshold** for better business alignment  
- Packaged the trained model, scaler, and threshold for production use  
- Deployed the solution as a **user-friendly Streamlit web app**

---

## 📊 Key Features

✔ Real-time churn probability prediction  
✔ Business-friendly risk classification (High / Low risk)  
✔ Clean, professional dashboard UI  
✔ Modular and production-ready code structure  
✔ Easy to deploy and scale  

---

## 📥 Model Inputs

The model evaluates churn risk using employee attributes such as:

- Demographics (gender, senior status, dependents)  
- Tenure and monthly charges  
- Contract type and billing preferences  
- Internet and service usage features  

---

## 📤 Output for Decision Makers

- **Churn Probability (%)**
- **Actionable Risk Status**
  - ⚠️ High Risk — Retention intervention recommended  
  - ✅ Low Risk — Employee likely to stay  

This output is designed to be **immediately understandable by non-technical users**.

---

## 🧪 Model & Evaluation

- Algorithm: Supervised ML classifier  
- Feature Scaling: `StandardScaler`  
- Optimization focus: **Recall (churn = Yes)**  
- Deployment artifacts: model + scaler + threshold (`joblib`)  

This ensures the model prioritizes identifying employees who are truly at risk.

---

## 🖥️ Application Interface

- Built using **Streamlit**
- Custom dark theme with clear color grading
- Structured sections for better usability
- One-click churn analysis

Designed to resemble a **real internal HR analytics tool**.

---

## 🛠️ Tech Stack

- **Python**
- **NumPy, Scikit-learn**
- **Joblib**
- **Streamlit**
- **Custom CSS for UI styling**