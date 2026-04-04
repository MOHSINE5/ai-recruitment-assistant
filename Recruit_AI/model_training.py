# ============================================================
# model_training.py — Entraînement des modèles de classification
# Projet : Outil intelligent d'aide au recrutement
# Auteur : KHATTACH MOHSINE (Projet 27)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib

# ============================================================
# 1. Chargement et préparation des données
# ============================================================
print("=" * 60)
print("  ENTRAÎNEMENT DES MODÈLES DE RECRUTEMENT")
print("=" * 60)

df = pd.read_csv("recruitment_data.csv")

# Séparation des features et de la cible
X = df.drop('HiringDecision', axis=1)
y = df['HiringDecision']

print(f"\n📐 Features (X) : {X.shape[1]} colonnes — {list(X.columns)}")
print(f"🎯 Cible (y)    : {y.value_counts().to_dict()}")
print(f"⚠️  Déséquilibre : {y.value_counts()[0]} non-recrutés vs {y.value_counts()[1]} recrutés")

# Division en ensembles d'entraînement et de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Entraînement : {X_train.shape[0]} échantillons")
print(f"📊 Test         : {X_test.shape[0]} échantillons")

# ============================================================
# 2. Définition et entraînement des modèles
# ============================================================
modeles = {
    "Random Forest": RandomForestClassifier(
        class_weight='balanced', random_state=42
    ),
    "Régression Logistique": LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    )
}

resultats = {}

for nom, modele in modeles.items():
    print(f"\n{'─' * 60}")
    print(f"  🔧 Modèle : {nom}")
    print(f"{'─' * 60}")

    # Entraînement
    modele.fit(X_train, y_train)

    # Prédictions
    y_pred = modele.predict(X_test)

    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Stockage des résultats
    resultats[nom] = {
        'modele': modele,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

    # Affichage des résultats en français
    print(f"\n  📈 Exactitude (Accuracy)  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  🎯 Précision              : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  📡 Rappel (Recall)        : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  ⚖️  Score F1               : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"\n  🔢 Matrice de confusion :")
    print(f"       Prédit →     Non recruté    Recruté")
    print(f"       Réel ↓")
    print(f"       Non recruté    {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"       Recruté        {cm[1][0]:>6}        {cm[1][1]:>6}")

# ============================================================
# 3. Comparaison et sélection du meilleur modèle
# ============================================================
print(f"\n{'=' * 60}")
print("  📊 COMPARAISON DES MODÈLES")
print(f"{'=' * 60}")

# Tableau récapitulatif
print(f"\n  {'Modèle':<25} {'Accuracy':>10} {'Précision':>10} {'Rappel':>10} {'F1-Score':>10}")
print(f"  {'─' * 65}")
for nom, res in resultats.items():
    print(f"  {nom:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1']:>10.4f}")

# Sélection du meilleur modèle basée sur le F1-score
# (F1 est préférable avec des données déséquilibrées)
meilleur_nom = max(resultats, key=lambda k: resultats[k]['f1'])
meilleur = resultats[meilleur_nom]

print(f"\n{'=' * 60}")
print(f"  🏆 MEILLEUR MODÈLE : {meilleur_nom}")
print(f"{'=' * 60}")
print(f"\n  ➡️  Le modèle '{meilleur_nom}' est sélectionné car il obtient")
print(f"     le meilleur score F1 ({meilleur['f1']:.4f}).")
print(f"     Le F1-score est la métrique privilégiée ici car le dataset")
print(f"     est déséquilibré (1035 non-recrutés vs 465 recrutés).")
print(f"     Le F1 équilibre précision et rappel, ce qui est crucial")
print(f"     pour ne pas manquer de bons candidats (faux négatifs)")
print(f"     tout en évitant de recommander les mauvais (faux positifs).")

# ============================================================
# 4. Sauvegarde du meilleur modèle
# ============================================================
joblib.dump(meilleur['modele'], 'model.pkl')
print(f"\n  💾 Modèle sauvegardé : model.pkl")
print(f"\n{'=' * 60}")
print(f"  ✅ ENTRAÎNEMENT TERMINÉ")
print(f"{'=' * 60}")
