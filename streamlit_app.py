import streamlit as st
import joblib
import numpy as np
import os


# Interface utilisateur
st.title("📊 Prédiction de l'Éligibilité au Crédit")
st.write(
    "Interface Web Simple d'éligible ou non  du client à un crédit"
)

# Charger le modèle
model = joblib.load("logistic_regression_model.joblib")


# Barre latérale pour les entrées utilisateur
st.sidebar.header("📝 Informations du client")
age = st.sidebar.number_input("Âge du client", min_value=18, max_value=100, value=30)
revenu = st.sidebar.number_input("Revenu mensuel (€)", min_value=0, value=3000)
historique_credit = st.sidebar.selectbox("Historique de crédit", ["Bon", "Moyen", "Mauvais"])
montant_loan = st.sidebar.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)

# Vérifier les incohérences
if montant_loan > revenu * 12:
    st.sidebar.warning("⚠️ Attention : Le montant du prêt dépasse 12 mois de revenu.")

# Prétraitement des données
historique_credit_map = {"Bon": 2, "Moyen": 1, "Mauvais": 0}
X_new = np.array([[age, revenu, historique_credit_map[historique_credit], montant_loan]])

# Prédiction
if st.sidebar.button("📊 Prédire"):
    prediction = model.predict(X_new)[0]
    statut = "GOOD" if prediction == 1 else "BAD"

    st.subheader("📌 Résultat de la Prédiction")
    st.write(f"🔍 **Statut prédit :** `{statut}`")

    if statut == "GOOD":
        st.success("✅ Le client est éligible au crédit !")
    else:
        st.error("❌ Le client n'est pas éligible au crédit.")

    # Affichage de la probabilité si disponible
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0][1]  # Probabilité d'être "GOOD"
        st.write(f"📊 **Probabilité d'éligibilité** : `{proba:.2%}`")
