import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# Configuration de base pour l'esthétique
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Création du répertoire s'il n'existe pas
os.makedirs("plots", exist_ok=True)

# 1. Chargement des données et séparation
df = pd.read_csv("recruitment_data.csv")
X = df.drop('HiringDecision', axis=1)
y = df['HiringDecision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entraînement des modèles (répliqué depuis model_training.py)
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# =========================================================
# GRAPHIQUE 1: Matrice de Confusion - Random Forest
# =========================================================
fig, ax = plt.subplots(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Non Recruté (0)', 'Recruté (1)'], 
            yticklabels=['Non Recruté (0)', 'Recruté (1)'], ax=ax,
            annot_kws={"size": 14})
ax.set_title("Matrice de Confusion\n(Random Forest - Meilleur Modèle)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Prédiction du Modèle', fontsize=12)
ax.set_ylabel('Valeur Réelle', fontsize=12)
plt.savefig("plots/matrice_confusion_rf.png")
plt.close()
print("✅ Généré : plots/matrice_confusion_rf.png")

# =========================================================
# GRAPHIQUE 2: Matrice de Confusion - Régression Logistique
# =========================================================
fig, ax = plt.subplots(figsize=(6, 5))
cm_lr = confusion_matrix(y_test, lr_model.predict(X_test))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Recruté (0)', 'Recruté (1)'], 
            yticklabels=['Non Recruté (0)', 'Recruté (1)'], ax=ax,
            annot_kws={"size": 14})
ax.set_title("Matrice de Confusion\n(Régression Logistique)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Prédiction du Modèle', fontsize=12)
ax.set_ylabel('Valeur Réelle', fontsize=12)
plt.savefig("plots/matrice_confusion_lr.png")
plt.close()
print("✅ Généré : plots/matrice_confusion_lr.png")

# =========================================================
# GRAPHIQUE 3: Importance des Variables (Feature Importance)
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)
features = X.columns
couleurs_bar = ['#34495E'] * (len(indices) - 3) + ['#2ECC71'] * 3 # Met en évidence le top 3

ax.barh(range(len(indices)), importances[indices], color=couleurs_bar, align='center', edgecolor='none')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices], fontsize=11)
ax.set_title("Importance des Variables dans la Décision (Random Forest)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Poids / Importance relative dans le modèle", fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("plots/importance_variables_rf.png")
plt.close()
print("✅ Généré : plots/importance_variables_rf.png")

# =========================================================
# GRAPHIQUE 4: Courbe ROC comparative (Performance)
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Calcul ROC pour RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, color='#2ECC71', lw=2.5, 
        label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Calcul ROC pour LR
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax.plot(fpr_lr, tpr_lr, color='#3498DB', lw=2, linestyle='-.', 
        label=f'Régression Logistique (AUC = {roc_auc_lr:.2f})')

# Diagonale parfaite (Aléatoire)
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Prédiction Aléatoire')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Taux de Faux Positifs (Spécificité inverse)', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (Sensibilité / Rappel)', fontsize=12)
ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc="lower right", fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.savefig("plots/courbe_roc_comparaison.png")
plt.close()
print("✅ Généré : plots/courbe_roc_comparaison.png")
