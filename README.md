# Telco Customer Churn Prediction with Data Quality Validation

## Project Overview

Customer churn prediction is a key problem for telecom companies because losing customers directly impacts revenue.  
The goal of this project is to build a **machine learning pipeline capable of predicting customer churn**, while ensuring **data quality, model validation, and fairness analysis**.

This project focuses not only on building a predictive model, but also on implementing **data validation and quality control processes** before training the model.

---

# Dataset

The dataset used in this project is the **Telco Customer Churn dataset**, which contains information about telecom customers including:

- demographic information
- services subscribed
- billing information
- contract details
- churn status

Each row represents a customer and the target variable indicates whether the customer **churned (Yes) or not (No)**.

---

# Project Structure
telco-churn-qc-project
│
├── telco_churn_project.ipynb
├── data_quality_checks.yaml
├── README.md
└── data
└── telco_customer_churn.csv

---

# Phase 1 — Data Quality Control (Data QC)

Before training any model, the dataset was validated to ensure the quality of the data.

This phase includes:

• Data profiling  
• Detection of missing values  
• Detection of duplicate records  
• Schema validation using **Pandera**  
• Outlier detection using statistical analysis

This step ensures that the dataset used for training is reliable and consistent.

---

# Phase 2 — Feature Engineering Validation

The preprocessing pipeline was carefully designed and validated.

This phase includes:

• Creation of a preprocessing pipeline  
• Encoding categorical variables  
• Scaling numerical variables  
• Unit tests to validate transformation functions  
• Verification of **data leakage prevention**

The pipeline ensures that **data transformations are applied consistently during training and evaluation**.

---

# Phase 3 — Model Validation

The machine learning model was evaluated using several techniques to ensure robust performance.

Validation techniques include:

• Train-test split  
• **Stratified cross-validation** to ensure stable performance  
• Confusion matrix analysis  
• Precision, Recall, and F1-score evaluation  
• Fairness analysis across demographic groups

---

# Model Performance

Results obtained on the test set:

| Metric | Value |
|------|------|
| Accuracy | ~0.80 |
| Precision (Churn) | ~0.64 |
| Recall (Churn) | ~0.56 |
| F1 Score (Churn) | ~0.60 |

The model performs well in identifying non-churn customers, while churn prediction remains more challenging due to class imbalance.

---

# Fairness Analysis

Model performance was also analyzed across different demographic groups:

### Gender

| Group | Accuracy |
|------|------|
| Female | ~0.79 |
| Male | ~0.81 |

No significant bias was observed between genders.

### Senior Citizens

| Group | Accuracy |
|------|------|
| Non-senior | ~0.81 |
| Senior | ~0.73 |

A slight performance difference was observed for senior customers, which may require further investigation.

---

# Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Pandera  
Matplotlib  
Seaborn  
Jupyter Notebook / VS Code

---

# Key Learnings

This project demonstrates the importance of:

• Data quality validation before modeling  
• Proper feature engineering pipelines  
• Robust model evaluation techniques  
• Fairness considerations in machine learning systems

---

# Possible Improvements

Future improvements could include:

• Testing more advanced models (Random Forest, Gradient Boosting)  
• Hyperparameter optimization  
• Handling class imbalance with techniques such as **SMOTE**  
• Deployment of the model in a production pipeline

---

# Author


## 0. Compréhension du Dataset
- Chargement des données
- Description des variables
- Identification de la variable cible

## 1. Contrôle Qualité des Données (Data QC)

### 1.1 Profilage des données
### 1.2 Détection des valeurs manquantes
### 1.3 Détection des doublons
### 1.4 Validation du schéma
### 1.5 Analyse des distributions
### 1.6 Détection des outliers

## 2. Validation de l’Ingénierie des Caractéristiques

### 2.1 Nettoyage et préparation des données
### 2.2 Feature engineering
### 2.3 Encodage des variables catégorielles
### 2.4 Normalisation / scaling
### 2.5 Tests unitaires du pipeline
### 2.6 Vérification du data leakage

## 3. Validation du Modèle

### 3.1 Séparation Train / Validation / Test
### 3.2 Entraînement du modèle baseline
### 3.3 Cross-validation stratifiée
### 3.4 Analyse de la matrice de confusion
### 3.5 Calcul des métriques (Precision, Recall, F1, AUC)
### 3.6 Analyse d’équité (Fairness)

## 4. Monitoring et Drift

### 4.1 Simulation de nouvelles données
### 4.2 Détection de data drift
### 4.3 Test statistique Kolmogorov-Smirnov
### 4.4 Définition des seuils d’alerte

## 5. Tableau de Bord des Métriques

### 5.1 Comparaison Train / Validation / Test
### 5.2 Visualisation des performances du modèle

## 6. Rapport Final

### 6.1 Limites du modèle
### 6.2 Risques et biais
### 6.3 Recommandations
