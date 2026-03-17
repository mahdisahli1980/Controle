# Pipeline de Score de Crédit – Zero Defect

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
