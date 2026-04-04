# ============================================================
# app.py — Interface Streamlit pour l'aide au recrutement
# Projet : Outil intelligent d'aide au recrutement
# Auteur : KHATTACH MOHSINE (Projet 27)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration de la page ---
st.set_page_config(
    page_title="Outil Intelligent d'Aide au Recrutement",
    page_icon="🇲🇦",
    layout="centered"
)

# ============================================================
# Style CSS personnalisé
# ============================================================
st.markdown("""
<style>
    /* Hide deploy button */
    .stDeployButton {
        visibility: hidden;
    }

    /* Hide hamburger menu (three dots) */
    #MainMenu {
        visibility: hidden;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    /* Optional: Hide header */
    header {
        visibility: hidden;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .result-negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .confidence-text {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# En-tête principal
# ============================================================
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🇲🇦 Outil Intelligent d'Aide au Recrutement")
st.markdown("*Système de prédiction basé sur le Machine Learning pour le recrutement*")
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ============================================================
# Chargement du modèle
# ============================================================
@st.cache_resource
def charger_modele():
    """Charge le modèle de prédiction sauvegardé."""
    return joblib.load('model.pkl')

try:
    modele = charger_modele()
except FileNotFoundError:
    st.error("⚠️ Le fichier `model.pkl` est introuvable. "
             "Veuillez d'abord exécuter `python model_training.py` pour entraîner le modèle.")
    st.stop()

# ============================================================
# Formulaire de saisie des données du candidat
# ============================================================
st.markdown("### 📝 Informations du candidat")
st.markdown("Remplissez les informations ci-dessous pour analyser le profil du candidat.")

# Organisation en colonnes pour un meilleur affichage
col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "🎂 Âge du candidat",
        min_value=26, max_value=60, value=35, step=1
    )

    genre_label = st.selectbox(
        "👤 Genre",
        options=["Femme (0)", "Homme (1)"]
    )
    genre = int(genre_label.split("(")[1].replace(")", ""))

    education_label = st.selectbox(
        "🎓 Niveau d'éducation",
        options=["Bac (1)", "Licence (2)", "Master (3)", "Doctorat (4)"]
    )
    education = int(education_label.split("(")[1].replace(")", ""))

    experience = st.slider(
        "💼 Années d'expérience",
        min_value=0, max_value=15, value=5, step=1
    )

    entreprises = st.slider(
        "🏢 Nombre d'entreprises précédentes",
        min_value=1, max_value=5, value=2, step=1
    )

with col2:
    distance = st.number_input(
        "📍 Distance domicile–entreprise (km)",
        min_value=0.0, max_value=100.0, value=15.0, step=0.5,
        format="%.1f"
    )

    score_entretien = st.slider(
        "🗣️ Score d'entretien",
        min_value=0, max_value=100, value=60, step=1
    )

    score_competences = st.slider(
        "💻 Score de compétences techniques",
        min_value=0, max_value=100, value=60, step=1
    )

    score_personnalite = st.slider(
        "🧠 Score de personnalité",
        min_value=0, max_value=100, value=60, step=1
    )

    strategie_label = st.selectbox(
        "📊 Stratégie de recrutement",
        options=["Agressive (1)", "Modérée (2)", "Conservative (3)"]
    )
    strategie = int(strategie_label.split("(")[1].replace(")", ""))

st.divider()

# ============================================================
# Bouton de prédiction
# ============================================================
if st.button("🔍 Analyser le candidat", type="primary", use_container_width=True):

    # Préparation des données d'entrée
    donnees_candidat = pd.DataFrame([{
        'Age': age,
        'Gender': genre,
        'EducationLevel': education,
        'ExperienceYears': experience,
        'PreviousCompanies': entreprises,
        'DistanceFromCompany': distance,
        'InterviewScore': score_entretien,
        'SkillScore': score_competences,
        'PersonalityScore': score_personnalite,
        'RecruitmentStrategy': strategie
    }])

    # Prédiction
    prediction = modele.predict(donnees_candidat)[0]
    probabilites = modele.predict_proba(donnees_candidat)[0]
    confiance = probabilites[prediction] * 100

    # Affichage du résultat
    st.markdown("### 📊 Résultat de l'analyse")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box result-positive">
            <h2>✅ Candidat recommandé pour le recrutement</h2>
            <p class="confidence-text">Confiance : {confiance:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        # st.balloons()
    else:
        st.markdown(f"""
        <div class="result-box result-negative">
            <h2>❌ Candidat non recommandé</h2>
            <p class="confidence-text">Confiance : {confiance:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Barre de probabilités
    st.markdown("#### 📈 Probabilités détaillées")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Non recruté", f"{probabilites[0]*100:.1f}%")
    with prob_col2:
        st.metric("Recruté", f"{probabilites[1]*100:.1f}%")

    st.progress(probabilites[1])

    # Tableau récapitulatif des entrées
    st.markdown("#### 📋 Récapitulatif du profil")

    # Mapping des labels pour l'affichage
    labels_education = {1: 'Bac', 2: 'Licence', 3: 'Master', 4: 'Doctorat'}
    labels_genre = {0: 'Femme', 1: 'Homme'}
    labels_strategie = {1: 'Agressive', 2: 'Modérée', 3: 'Conservative'}

    recap = pd.DataFrame({
        'Critère': [
            'Âge', 'Genre', "Niveau d'éducation", "Années d'expérience",
            'Entreprises précédentes', 'Distance domicile-entreprise',
            "Score d'entretien", 'Score compétences techniques',
            'Score personnalité', 'Stratégie de recrutement'
        ],
        'Valeur': [
            f"{age} ans",
            labels_genre[genre],
            labels_education[education],
            f"{experience} ans",
            str(entreprises),
            f"{distance:.1f} km",
            f"{score_entretien}/100",
            f"{score_competences}/100",
            f"{score_personnalite}/100",
            labels_strategie[strategie]
        ]
    })

    st.dataframe(recap, use_container_width=True, hide_index=True)
