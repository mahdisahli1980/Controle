import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ks_2samp

st.set_page_config(page_title="Telco Churn QC Dashboard", layout="wide")

# =========================================================
# Fonctions utilitaires
# =========================================================
def clean_telco(df):
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def standardize_binary_labels(series):
    mapping = {
        "Yes": 1, "No": 0,
        "yes": 1, "no": 0,
        "Y": 1, "N": 0,
        "True": 1, "False": 0,
        True: 1, False: 0,
        "1": 1, "0": 0,
        1: 1, 0: 0
    }
    s = series.copy()
    s = s.map(lambda x: mapping[x] if x in mapping else x)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def count_outliers_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def plot_histogram(df, column, title):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.hist(pd.to_numeric(df[column], errors="coerce").dropna(), bins=25)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(column, fontsize=9)
    ax.set_ylabel("Fréquence", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    st.pyplot(fig, clear_figure=True)


def plot_missing_values(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "Aucune valeur manquante", ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        ax.bar(missing.index, missing.values)
        ax.set_title("Valeurs manquantes", fontsize=10)
        ax.set_ylabel("Nombre", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    st.pyplot(fig, clear_figure=True)


def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm)
    ax.set_title("Matrice de confusion", fontsize=10)
    ax.set_xlabel("Prédit", fontsize=9)
    ax.set_ylabel("Réel", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=8)
    ax.set_yticklabels(["0", "1"], fontsize=8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10)

    st.pyplot(fig, clear_figure=True)


def fairness_table(pred_df, sensitive_col):
    rows = []
    for group in pred_df[sensitive_col].dropna().unique():
        sub = pred_df[pred_df[sensitive_col] == group].copy()
        if len(sub) == 0:
            continue

        rows.append({
            sensitive_col: group,
            "accuracy": round(accuracy_score(sub["y_true"], sub["y_pred"]), 4),
            "precision": round(precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0), 4),
            "recall": round(recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0), 4),
            "f1": round(f1_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0), 4),
            "selection_rate_1": round((sub["y_pred"] == 1).mean(), 4)
        })

    return pd.DataFrame(rows)


def plot_fairness_bar(fair_df, group_col, metric):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(fair_df[group_col].astype(str), fair_df[metric])
    ax.set_title(f"{metric} par {group_col}", fontsize=10)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_xlabel(group_col, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    st.pyplot(fig, clear_figure=True)


def back_to_menu():
    if st.button("Retour au menu principal"):
        st.session_state.page = "menu"
        st.rerun()


# =========================================================
# Chargement automatique des fichiers
# =========================================================
raw_file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
pred_file_path = "predictions_dashboard.csv"
new_data_file_path = "new_data_drift.csv"

try:
    df = pd.read_csv(raw_file_path)
    df = clean_telco(df)
    raw_data_available = True
except Exception as e:
    raw_data_available = False
    raw_error = str(e)

try:
    pred_df = pd.read_csv(pred_file_path)
    pred_data_available = True
except Exception as e:
    pred_data_available = False
    pred_error = str(e)

try:
    new_df = pd.read_csv(new_data_file_path)
    new_df = clean_telco(new_df)
    drift_available = True
except Exception:
    drift_available = False

if not raw_data_available:
    st.error(f"Impossible de charger le fichier principal : {raw_file_path}")
    st.code(raw_error)
    st.stop()


# =========================================================
# Préparation prédictions
# =========================================================
if pred_data_available:
    required_pred_cols = {"y_true", "y_pred"}
    if required_pred_cols.issubset(pred_df.columns):
        pred_df["y_true"] = standardize_binary_labels(pred_df["y_true"])
        pred_df["y_pred"] = standardize_binary_labels(pred_df["y_pred"])
        pred_df = pred_df.dropna(subset=["y_true", "y_pred"]).copy()
        pred_df["y_true"] = pred_df["y_true"].astype(int)
        pred_df["y_pred"] = pred_df["y_pred"].astype(int)
    else:
        pred_data_available = False
        pred_error = "Le fichier predictions_dashboard.csv doit contenir au moins : y_true, y_pred"


# =========================================================
# Navigation
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "menu"


# =========================================================
# MENU PRINCIPAL
# =========================================================
if st.session_state.page == "menu":
    st.title("Pipeline Zero Defect")
    st.markdown("### Navigation du projet")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("0. Vue générale du dataset", use_container_width=True):
            st.session_state.page = "overview"
            st.rerun()

        if st.button("1. Phase 1 - Contrôle Qualité des Données", use_container_width=True):
            st.session_state.page = "phase1"
            st.rerun()

        if st.button("2. Phase 2 - Validation de l’Ingénierie des Caractéristiques", use_container_width=True):
            st.session_state.page = "phase2"
            st.rerun()

    with col2:
        if st.button("3. Phase 3 - Validation du Modèle", use_container_width=True):
            st.session_state.page = "phase3"
            st.rerun()

        if st.button("4. Phase 4 - Monitoring et Drift", use_container_width=True):
            st.session_state.page = "phase4"
            st.rerun()

        if st.button("5. Conclusion", use_container_width=True):
            st.session_state.page = "conclusion"
            st.rerun()


# =========================================================
# VUE GÉNÉRALE
# =========================================================
elif st.session_state.page == "overview":
    st.title("0. Vue générale du dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Doublons", int(df.duplicated().sum()))

    st.write("Aperçu des données")
    st.dataframe(df.head())

    back_to_menu()


# =========================================================
# PHASE 1
# =========================================================
elif st.session_state.page == "phase1":
    st.title("1. Contrôle Qualité des Données")

    st.markdown("""
**Objectif :** vérifier la qualité des données avant toute modélisation.
Cette section présente les valeurs manquantes, les distributions et les outliers.
""")

    tab1, tab2, tab3 = st.tabs(["Valeurs manquantes", "Distribution", "Outliers"])

    with tab1:
        plot_missing_values(df)

    with tab2:
        numeric_choices = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
        if numeric_choices:
            selected_col = st.selectbox("Choisir une variable numérique", numeric_choices, key="phase1_dist")
            plot_histogram(df, selected_col, f"Distribution de {selected_col}")

    with tab3:
        outlier_results = []
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in df.columns:
                outlier_results.append({
                    "variable": col,
                    "nombre_outliers": count_outliers_iqr(df[col])
                })
        st.dataframe(pd.DataFrame(outlier_results))

    st.markdown("""
**Résumé :**
- cette phase permet de détecter rapidement les problèmes de qualité,
- les valeurs manquantes et les outliers sont identifiés,
- les distributions sont vérifiées avant la suite du pipeline.
""")

    back_to_menu()


# =========================================================
# PHASE 2
# =========================================================
elif st.session_state.page == "phase2":
    st.title("2. Validation de l’Ingénierie des Caractéristiques")

    st.markdown("""
**Objectif :** vérifier que les transformations appliquées aux données sont cohérentes
et qu’aucune fuite de données n’est introduite dans le pipeline.
""")

    st.markdown("""
### Éléments validés
- nettoyage et préparation des colonnes,
- encodage des variables catégorielles,
- standardisation / prétraitement,
- séparation correcte des jeux de données,
- absence de data leakage entre variables explicatives et cible.
""")

    st.markdown("""
### Remarque
Cette phase est surtout validée dans le notebook à travers :
- le pipeline de prétraitement,
- la séparation train / validation / test,
- les transformations appliquées avant l'entraînement du modèle.
""")

    st.markdown("""
**Conclusion :**
les étapes de transformation ont été intégrées dans un pipeline structuré,
ce qui garantit une préparation cohérente des données avant la modélisation.
""")

    back_to_menu()


# =========================================================
# PHASE 3
# =========================================================
elif st.session_state.page == "phase3":
    st.title("3. Validation du Modèle")

    st.markdown("""
**Modèle évalué : Logistic Regression**

**Objectif :**  
Évaluer les performances du modèle, vérifier sa robustesse et analyser l'équité entre groupes.
""")

    if pred_data_available:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(pred_df['y_true'], pred_df['y_pred']):.4f}")
        c2.metric("Precision", f"{precision_score(pred_df['y_true'], pred_df['y_pred'], pos_label=1, zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(pred_df['y_true'], pred_df['y_pred'], pos_label=1, zero_division=0):.4f}")
        c4.metric("F1-score", f"{f1_score(pred_df['y_true'], pred_df['y_pred'], pos_label=1, zero_division=0):.4f}")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Matrice de confusion")
            plot_conf_matrix(pred_df["y_true"], pred_df["y_pred"])

        with col_right:
            st.markdown("### Analyse d'équité")
            fairness_options = []
            if "gender" in pred_df.columns:
                fairness_options.append("gender")
            if "SeniorCitizen" in pred_df.columns:
                fairness_options.append("SeniorCitizen")

            if fairness_options:
                group_col = st.selectbox("Choisir la variable sensible", fairness_options, key="phase3_group")
                fair_df = fairness_table(pred_df, group_col)
                st.dataframe(fair_df, use_container_width=True)

                metric_choice = st.selectbox(
                    "Choisir la métrique à visualiser",
                    ["accuracy", "precision", "recall", "f1", "selection_rate_1"],
                    key="phase3_metric"
                )
                plot_fairness_bar(fair_df, group_col, metric_choice)
            else:
                st.info("Aucune colonne sensible trouvée dans predictions_dashboard.csv.")

        st.markdown("""
**Conclusion :**
le modèle a été évalué avec plusieurs métriques complémentaires,
et une analyse d’équité a été réalisée pour vérifier l’absence de biais majeur entre groupes.
""")
    else:
        st.warning(f"Impossible de charger le fichier : {pred_file_path}")
        st.code(pred_error)

    back_to_menu()


# =========================================================
# PHASE 4
# =========================================================
elif st.session_state.page == "phase4":
    st.title("4. Monitoring et Drift")

    st.markdown("""
**Objectif :** détecter un changement de distribution entre les données initiales
et un nouveau jeu de données simulé après déploiement.
""")

    if drift_available:
        drift_results = []

        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in df.columns and col in new_df.columns:
                old_series = pd.to_numeric(df[col], errors="coerce").dropna()
                new_series = pd.to_numeric(new_df[col], errors="coerce").dropna()

                if len(old_series) > 0 and len(new_series) > 0:
                    ks_stat, p_value = ks_2samp(old_series, new_series)
                    drift_results.append({
                        "variable": col,
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_value), 6),
                        "drift_detected": "Oui" if p_value < 0.05 else "Non"
                    })

        drift_df = pd.DataFrame(drift_results)
        st.dataframe(drift_df, use_container_width=True)

        compare_choices = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns and c in new_df.columns]

        if compare_choices:
            compare_col = st.selectbox(
                "Comparer visuellement une variable pour le drift",
                compare_choices,
                key="phase4_compare"
            )

            fig, ax = plt.subplots(figsize=(4.4, 2.8))
            ax.hist(pd.to_numeric(df[compare_col], errors="coerce").dropna(), bins=25, alpha=0.6, label="Initial")
            ax.hist(pd.to_numeric(new_df[compare_col], errors="coerce").dropna(), bins=25, alpha=0.6, label="Nouveau")
            ax.set_title(f"Comparaison - {compare_col}", fontsize=10)
            ax.set_xlabel(compare_col, fontsize=9)
            ax.set_ylabel("Fréquence", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.legend(fontsize=8)
            st.pyplot(fig, clear_figure=True)

        st.markdown("""
**Interprétation :**
si la p-value est inférieure à 0.05, on considère qu’un drift significatif est détecté.
""")
    else:
        st.info("Le fichier new_data_drift.csv n'a pas été trouvé. La section drift reste désactivée.")

    back_to_menu()


# =========================================================
# CONCLUSION
# =========================================================
elif st.session_state.page == "conclusion":
    st.title("5. Conclusion")

    st.markdown("""
Ce projet a permis de construire un pipeline structuré autour de quatre dimensions essentielles :

- **Contrôle qualité des données**
- **Validation de l’ingénierie des caractéristiques**
- **Validation du modèle**
- **Monitoring et détection du drift**

L’ensemble permet de présenter un système de scoring plus fiable, plus explicable
et plus robuste face aux changements futurs des données.
""")

    back_to_menu()