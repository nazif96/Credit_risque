import streamlit as st
import joblib
import numpy as np
import os

# Charger le modèle avec gestion d'erreur
model_path = "log_reg_model.joblib"
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

# Vérifier les incohérences
if montant_loan > revenu * 12:
    st.sidebar.warning("⚠️ Attention : Le montant du prêt dépasse 12 mois de revenu.")

# Sélection des caractéristiques catégoriques
job = st.sidebar.selectbox("🛠 Emploi", ["Qualifie", "Non qualifie", "Hautement_qualifie_Indépendant", "Chomeur"])
credit_history = st.sidebar.selectbox("💳 Historique de crédit", [
    "Crédit_Autres_crédits_critique", "Crédits_existants_remboursés", "Paiements_retardés_auparavant",
    "Aucun_crédit", "crédits_payés"
])
other_debtors = st.sidebar.selectbox("👥 Autres débiteurs / garants", ["Aucun", "Garant", "Co-emprunteur"])
housing = st.sidebar.selectbox("🏠 Logement", ["Propriétaire", "Logé_gratuitement", "Locataire"])
saving_status = st.sidebar.selectbox("💰 Épargne", ["Pas_dépargne", "Moins_de_100", "Entre 100_et_500", "Entre 500_et_1000", "Plus_de_1000"])
credit_purpose = st.sidebar.selectbox("🎯 Objet du crédit", [
    "Radio_TV", "Education", "Mobilier_ou_Equipement", "Voiture_neuve", "Voiture_occasion", "Affaires",
    "Appareil_electromenager", "Reparations", "Autres", "Reconversion"
])

# Mapping des valeurs catégoriques en numériques
job_map = {
    "Qualifie": 0, "Non qualifie": 1, "Hautement_qualifie_Indépendant": 2, "Chomeur": 3
}
credit_history_map = {
    "Crédit_Autres_crédits_critique": 0, "Crédits_existants_remboursés": 1, "Paiements_retardés_auparavant": 2,
    "Aucun_crédit": 3, "crédits_payés": 4
}
other_debtors_map = {"Aucun": 0, "Garant": 1, "Co-emprunteur": 2}
housing_map = {"Propriétaire": 0, "Logé_gratuitement": 1, "Locataire": 2}
saving_status_map = {
    "Pas_dépargne": 0, "Moins_de_100": 1, "Entre 100_et_500": 2, "Entre 500_et_1000": 3, "Plus_de_1000": 4
}
credit_purpose_map = {
    "Radio_TV": 0, "Education": 1, "Mobilier_ou_Equipement": 2, "Voiture_neuve": 3, "Voiture_occasion": 4,
    "Affaires": 5, "Appareil_electromenager": 6, "Reparations": 7, "Autres": 8, "Reconversion": 9
}

# Création de la matrice d'entrée
X_new = np.array([[
    age, revenu, montant_loan,
    job_map[job], credit_history_map[credit_history], other_debtors_map[other_debtors],
    housing_map[housing], saving_status_map[saving_status], credit_purpose_map[credit_purpose]
]], dtype=float)

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
