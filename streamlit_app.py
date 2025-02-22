import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("logistic_regression_model.joblib")

# Interface utilisateur
st.title("Prédiction de l'Éligibilité au Crédit")

# Entrée des informations client
age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
revenu = st.number_input("Revenu mensuel (€)", min_value=0, value=3000)
historique_credit = st.selectbox("Historique de crédit", ["Bon", "Moyen", "Mauvais"])
montant_loan = st.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)

# Prétraitement des données pour correspondre au format du modèle
historique_credit_map = {"Bon": 2, "Moyen": 1, "Mauvais": 0}
X_new = np.array([[age, revenu, historique_credit_map[historique_credit], montant_loan]])

# Faire la prédiction
if st.button("Prédire"):
    prediction = model.predict(X_new)[0]
    statut = "GOOD" if prediction == 1 else "BAD"
    
    st.write(f"🔍 **Statut du compte checking prédit** : `{statut}`")
    if statut == "GOOD":
        st.success("✅ Le client est éligible au crédit !")
    else:
        st.error("❌ Le client n'est pas éligible au crédit.")

