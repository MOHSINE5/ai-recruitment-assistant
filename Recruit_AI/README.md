# 🇲🇦 Recruit AI — Assistant Intelligent d'Aide au Recrutement

Ce projet est un outil d'aide à la décision pour le recrutement, utilisant le **Machine Learning** pour prédire si un candidat doit être recruté ou non en fonction de divers critères (expérience, éducation, scores d'entretien, etc.).

## 🚀 Fonctionnalités

- **Analyse Prédictive** : Utilisation d'un modèle de classification (Random Forest) pour évaluer les candidatures.
- **Score de Confiance** : Affiche la probabilité de réussite de la prédiction.
- **Interface Interactive** : Une application web moderne construite avec Streamlit.
- **Gestion du Déséquilibre** : Le modèle est entraîné en tenant compte du déséquilibre des classes dans les données de recrutement.
- **Récapitulatif Profil** : Affichage clair des critères saisis pour une validation rapide.

## 🛠️ Technologies Utilisées

- **Langage** : Python 3.x
- **Interface** : [Streamlit](https://streamlit.io/)
- **Machine Learning** : [Scikit-learn](https://scikit-learn.org/) (Random Forest, Logistic Regression)
- **Analyse de données** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **Persistance du modèle** : Joblib

## 📋 Prérequis

Assurez-vous d'avoir Python installé sur votre machine. Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## ⚙️ Utilisation

### 1. Entraîner le modèle
Si vous souhaitez réentraîner le modèle avec les données les plus récentes (`recruitment_data.csv`), lancez le script d'entraînement :

```bash
python model_training.py
```
Cela générera un fichier `model.pkl` contenant le meilleur modèle (sélectionné selon le score F1).

### 2. Lancer l'application
Pour démarrer l'interface utilisateur Streamlit, utilisez la commande suivante :

```bash
streamlit run app.py
```

L'application sera accessible localement (généralement à l'adresse `http://localhost:8501`).

## 📂 Structure du Projet

- `app.py` : Code principal de l'interface Streamlit.
- `model_training.py` : Script d'entraînement et de comparaison des modèles ML.
- `data_exploration.py` : Script d'analyse exploratoire des données (EDA).
- `model_evaluation_plots.py` : Script pour générer des graphiques d'évaluation.
- `recruitment_data.csv` : Jeu de données historique utilisé pour l'entraînement.
- `model.pkl` : Le modèle entraîné et sauvegardé.
- `requirements.txt` : Liste des bibliothèques Python requises.

## 👤 Auteur
- **KHATTACH MOHSINE** (Projet 27)
