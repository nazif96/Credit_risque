import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Charger le modèle
try:
    model = joblib.load("log_reg_model.joblib")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Définir manuellement les features attendues si elles sont connues
FEATURES_EXPECTED = [
    "age", "revenu", "historique_credit_Bon", "historique_credit_Mauvais", "montant_loan"
]  # Ajuste cette liste en fonction du modèle

# Interface utilisateur
st.title("📊 Prédiction de l'Éligibilité au Crédit")

with st.form("credit_form"):
    age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
    revenu = st.number_input("Revenu mensuel (€)", min_value=0, value=3000)
    historique_credit = st.selectbox("Historique de crédit", ["Bon", "Mauvais"])
    montant_loan = st.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)

    submit = st.form_submit_button("Prédire")

if submit:
    # Création du DataFrame
    data = {
        "age": [age],
        "revenu": [revenu],
        "historique_credit": [historique_credit],
        "montant_loan": [montant_loan]
    }

    X_new = pd.DataFrame(data)

    # **Encodage manuel des variables catégorielles (One-Hot Encoding)**
    X_new = pd.get_dummies(X_new, columns=["historique_credit"])

    # Ajouter les colonnes manquantes avec des valeurs par défaut (0 pour One-Hot Encoding)
    for col in FEATURES_EXPECTED:
        if col not in X_new.columns:
            X_new[col] = 0

    # Réordonner les colonnes selon celles attendues par le modèle
    X_new = X_new[FEATURES_EXPECTED]

    try:
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
