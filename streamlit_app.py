import streamlit as st
import joblib
import numpy as np
import os

# Charger le modèle avec gestion d'erreur
model_path = "logistic_regression_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("⚠️ Erreur : Le fichier du modèle n'a pas été trouvé.")
    st.stop()

# Vérifier le nombre de features attendues
expected_features = model.n_features_in_

# Interface utilisateur
st.title("📊 Prédiction de l'Éligibilité au Crédit")
st.write("Interface Web permettant de prédire si un client est éligible ou non à un crédit.")

# Barre latérale pour les entrées utilisateur
st.sidebar.header("📝 Informations du client")
age = st.sidebar.number_input("Âge du client", min_value=18, max_value=100, value=30)
revenu = st.sidebar.number_input("Revenu mensuel (€)", min_value=0, value=3000)
historique_credit = st.sidebar.selectbox("Historique de crédit", ["Bon", "Moyen", "Mauvais"])
montant_loan = st.sidebar.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)

# Vérifier les incohérences
if montant_loan > revenu * 12:
    st.sidebar.warning("⚠️ Attention : Le montant du prêt dépasse 12 mois de revenu.")

# Mapping des valeurs catégoriques
historique_credit_map = {"Bon": 2, "Moyen": 1, "Mauvais": 0}
X_new = np.array([[age, revenu, historique_credit_map[historique_credit], montant_loan]], dtype=float)

# Vérifier la structure des données
if X_new.shape[1] != expected_features:
    st.error(f"🚨 Erreur : Le modèle attend {expected_features} features, mais {X_new.shape[1]} ont été fournis.")
    st.stop()

# Appliquer un scaler si nécessaire
scaler_path = "scaler.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_new = scaler.transform(X_new)

# Bouton de prédiction
if st.sidebar.button("📊 Prédire"):
    # Prédiction de la classe (0 = BAD, 1 = GOOD)
    prediction = model.predict(X_new)[0]
    statut = "GOOD" if prediction == 1 else "BAD"

    # Prédiction de la probabilité d'acceptation
    prob_good = model.predict_proba(X_new)[0][1]  # Probabilité d'être "GOOD"

    # Affichage des résultats
    st.subheader("📌 Résultat de la Prédiction")
    st.write(f"🔍 **Statut prédit :** `{statut}`")
    st.write(f"📊 **Probabilité d'éligibilité au crédit** : `{prob_good:.2%}`")

    if statut == "GOOD":
        st.success("✅ Le client est éligible au crédit !")
    else:
        st.error("❌ Le client n'est pas éligible au crédit.")
