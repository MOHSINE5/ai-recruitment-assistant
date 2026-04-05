# 🇲🇦 Recruit AI — Assistant Intelligent d'Aide au Recrutement

Ce projet est un outil d'aide à la décision pour le recrutement, utilisant le **Machine Learning** pour prédire si un candidat doit être recruté ou non en fonction de divers critères (expérience, éducation, scores d'entretien, etc.).

## 🚀 Fonctionnalités Clés

- **📄 Analyse Intelligente de CV** : Nouveau module d'import permettant d'extraire automatiquement les informations clés depuis un CV au format **PDF** ou **DOCX** (Nom, Email, Âge, Expérience, Éducation, Score de Compétence).
- **Analyse Prédictive** : Utilisation d'un modèle de classification (Random Forest) pour évaluer la recommandation de recrutement.
- **Score de Confiance** : Affichage clair de la certitude du modèle pour chaque recommandation (en %).
- **Interface UI/UX Polie** : 
    - Séparation visuelle entre les données d'identité (non analysées) et les critères de profil.
    - Masquage des encodages Machine Learning techniques pour une expérience utilisateur fluide.
    - Tableau récapitulatif du profil interactif et rétractable via un expander.
- **Gestion du Déséquilibre** : Le modèle est entraîné en tenant compte du déséquilibre des classes dans les données historiques.

## 🛠️ Technologies Utilisées

- **Langage** : Python 3.x
- **Interface** : [Streamlit](https://streamlit.io/)
- **Machine Learning** : [Scikit-learn](https://scikit-learn.org/) (Random Forest, Logistic Regression)
- **Traitement de Documents** : PyPDF2, python-docx
- **Analyse de données** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **Persistance du modèle** : Joblib

## 📋 Prérequis

Assurez-vous d'avoir Python installé. Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## ⚙️ Utilisation

### 1. Préparer ou Entraîner le modèle
Si vous souhaitez réentraîner le modèle avec les données les plus récentes (`recruitment_data.csv`), lancez le script d'entraînement :

```bash
python model_training.py
```
Cela générera un fichier `model.pkl` contenant le meilleur modèle (sélectionné selon le score F1-Macro).

### 2. Lancer l'application
Pour démarrer l'interface utilisateur Streamlit, utilisez la commande suivante :

```bash
streamlit run app.py
```

L'application sera accessible localement (généralement à l'adresse `http://localhost:8501`).

## 📂 Structure du Projet

- `app.py` : Code principal de l'interface Streamlit (UI/UX et logique applicative).
- `cv_parser.py` : Moteur d'extraction et de parsing de données depuis les fichiers PDF/DOCX.
- `model_training.py` : Script d'entraînement et de comparaison automatique des modèles ML.
- `data_exploration.py` : Script d'analyse exploratoire des données (EDA).
- `model_evaluation_plots.py` : Script de génération des graphiques de performance.
- `recruitment_data.csv` : Jeu de données historique utilisé pour l'entraînement.
- `requirements.txt` : Liste des bibliothèques Python indispensables.

## 👤 Auteur
- **KHATTACH MOHSINE**
- *Projet 27 - Assistant Intelligent d'Aide au Recrutement*
