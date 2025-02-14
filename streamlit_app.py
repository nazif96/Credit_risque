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
montant_loan = st.sidebar.number_input("Montant du prêt demandé (€)", min_value=0, value=5000)
num_existing_credits = st.sidebar.number_input("Nombre de crédits existants", min_value=0, value=1)
checking_account_status = st.sidebar.selectbox("Statut du compte bancaire", [0, 1])

# Sélection des caractéristiques catégoriques
job = st.sidebar.selectbox("🛠 Emploi", ["Qualifie", "Non qualifie", "Hautement_qualifie_Indépendant"])
credit_history = st.sidebar.selectbox("💳 Historique de crédit", [
    "Crédit_Autres_crédits_critique", "Crédits_existants_remboursés", "Paiements_retardés_auparavant", "crédits_payés"
])
other_debtors = st.sidebar.selectbox("👥 Autres débiteurs / garants", ["Aucun", "Garant", "Co-emprunteur"])
housing = st.sidebar.selectbox("🏠 Logement", ["Propriétaire", "Logé_gratuitement"])
saving_status = st.sidebar.selectbox("💰 Épargne", ["Pas_dépargne", "Moins_de_100", "Entre 100_et_500", "Entre 500_et_1000", "Plus_de_1000"])
credit_purpose = st.sidebar.selectbox("🎯 Objet du crédit", [
    "Radio_TV", "Education", "Mobilier_ou_Equipement", "Voiture_neuve", "Voiture_occasion", "Affaires",
    "Appareil_electromenager", "Reparations", "Autres", "Reconversion"
])

# Encodage One-Hot
credit_history_features = [credit_history == f for f in ["Crédit_Autres_crédits_critique", "Crédits_existants_remboursés", "Paiements_retardés_auparavant", "crédits_payés"]]
credit_purpose_features = [credit_purpose == f for f in ["Radio_TV", "Education", "Mobilier_ou_Equipement", "Voiture_neuve", "Voiture_occasion", "Affaires", "Appareil_electromenager", "Reparations", "Autres", "Reconversion"]]
other_debtors_features = [other_debtors == "Co-emprunteur", other_debtors == "Garant"]
housing_features = [housing == "Logé_gratuitement"]
job_features = [job == "Hautement_qualifie_Indépendant", job == "Non qualifie"]
saving_status_features = [saving_status == f for f in ["Entre 500_et_1000", "Moins_de_100", "Pas_dépargne", "Plus_de_1000"]]

# Création de la matrice d'entrée
X_new = np.array([
    [age, revenu, montant_loan, num_existing_credits, checking_account_status] +
    credit_history_features + credit_purpose_features + other_debtors_features +
    housing_features + job_features + saving_status_features
], dtype=float)

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
    prediction = model.predict(X_new)[0]
    prob_good = model.predict_proba(X_new)[0][1]  # Probabilité d'être "GOOD"
    statut = "GOOD" if prediction == 1 else "BAD"
    
    st.subheader("📌 Résultat de la Prédiction")
    st.write(f"🔍 **Statut prédit :** `{statut}`")
    st.write(f"📊 **Probabilité d'éligibilité au crédit** : `{prob_good:.2%}`")

    if statut == "GOOD":
        st.success("✅ Le cSlient est éligible au crédit !")
    else:
        st.error("❌ Le client n'est pas éligible au crédit.")
