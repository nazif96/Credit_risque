import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Charger le modèle et le préprocesseur (si applicable)
try:
    model = joblib.load("log_reg_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")  # Charger le pipeline de prétraitement s'il existe
    feature_names = joblib.load("feature_names.joblib")  # Charger les noms des colonnes utilisées à l'entraînement
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle ou des preprocessors : {e}")
    st.stop()

# Interface utilisateur
st.title("📊 Prédiction de l'Éligibilité au Crédit")

with st.form("credit_form"):
    age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
    revenu = st.number_input("Revenu mensuel (€)", min_value=0, value=3000)
    historique_credit = st.selectbox("Historique de crédit", ["Bon", "Mauvais"])
    montant_loan = st.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)

    submit = st.form_submit_button("Prédire")

if submit:
    # Créer un dictionnaire avec les données de l'utilisateur
    data = {
        "age": [age],
        "revenu": [revenu],
        "historique_credit": [historique_credit],
        "montant_loan": [montant_loan]
    }
    
    X_new = pd.DataFrame(data)

    try:
        # Appliquer le prétraitement si nécessaire
        if "preprocessor" in locals():
            X_new = preprocessor.transform(X_new)

        # Convertir en DataFrame avec les colonnes attendues
        X_new = pd.DataFrame(X_new, columns=preprocessor.get_feature_names_out())

        # Vérifier si des colonnes sont manquantes
        missing_cols = set(feature_names) - set(X_new.columns)
        if missing_cols:
            st.warning(f"Colonnes manquantes ajoutées avec des valeurs par défaut : {missing_cols}")
            for col in missing_cols:
                X_new[col] = 0  # Valeur par défaut (ajuster selon le type des variables)

        # Réordonner les colonnes selon le modèle
        X_new = X_new[feature_names]

        # Vérifier que le nombre de features correspond
        if X_new.shape[1] != model.n_features_in_:
            st.error(f"Erreur : Le modèle attend {model.n_features_in_} features, mais {X_new.shape[1]} ont été fournies.")
            st.stop()

        # Prédiction
        prediction = model.predict(X_new)[0]
        statut = "GOOD" if prediction == 1 else "BAD"

        st.write(f"🔍 **Statut du compte checking prédit** : `{statut}`")
        if statut == "GOOD":
            st.success("✅ Le client est éligible au crédit !")
        else:
            st.error("❌ Le client n'est pas éligible au crédit.")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
