# Recidivism Risk Prediction with Explainable AI  

This project explores **recidivism risk prediction** using the [COMPAS two-year dataset](https://github.com/propublica/compas-analysis), focusing on **explainable AI (XAI)** techniques. The case study centers on Malik Johnson, a 27-year-old with one prior felony, who was flagged as low risk by the model.  

Our goal is not just to build a predictive model, but to **interpret its decisions** using tools like **LIME, SHAP, and Anchors**.  

---

## Dataset  

- **Source:** [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis)  
- **File:** `compas-scores-two-years.csv`  
- **Target:** `two_year_recid` (whether an individual reoffended within two years)  
- **Features used:**  
  - Age  
  - Sex  
  - Race  
  - Priors count  
  - Juvenile felony/misdemeanor/other counts  
  - Charge degree (felony or misdemeanor)  

---

##  Model  

We use a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) trained on the processed dataset.  


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)

---

## How to Run  
1. Clone the repo  
2. Run the notebook
