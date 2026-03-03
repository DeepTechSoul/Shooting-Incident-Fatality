# 🔎 Classifying Shooting Incident Fatality

A classification model that analyzes historical shooting incident data to predict fatality outcomes, providing law enforcement agencies with data-driven insights for targeted intervention and resource allocation.

## 🔍 Problem Statement
Understanding patterns in shooting incidents can help law enforcement agencies identify high-risk situations before they escalate. This model classifies incidents by fatality likelihood based on historical data.

## 🛠️ Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn
- **Visualization** — Matplotlib, Seaborn
- **Deployment** — Streamlit

## 📊 Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## ⚙️ Key Features Analyzed
- Time and location of incident
- Victim and perpetrator demographics
- Type of weapon involved
- Jurisdiction and precinct data

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run appfatality.py
```

## 📁 Project Structure
```
├── Classifying Shooting Incident Fatality.ipynb    # Full analysis
├── appfatality.py                                  # Streamlit web app
```

## 📌 Key Insights
- Time of day and location type are strong predictors of fatality outcome
- Certain weapon types correlate strongly with fatal incidents
- Model achieved strong precision-recall balance to minimize false negatives

---
*Project completed as part of Data Science internship at Spinnaker Analytics LLC (2024–2025)*
