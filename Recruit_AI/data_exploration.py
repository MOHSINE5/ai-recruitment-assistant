# ============================================================
# data_exploration.py — Exploration des données de recrutement
# Projet : Outil intelligent d'aide au recrutement
# Auteur : KHATTACH MOHSINE (Projet 27)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration de l'affichage ---
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# --- Création du dossier de sauvegarde des graphiques ---
os.makedirs("plots", exist_ok=True)

# ============================================================
# 1. Chargement des données
# ============================================================
print("=" * 60)
print("  EXPLORATION DES DONNÉES DE RECRUTEMENT")
print("=" * 60)

df = pd.read_csv("recruitment_data.csv")

# --- Forme du dataset ---
print(f"\n📐 Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Colonnes : {list(df.columns)}")

# --- Statistiques descriptives ---
print("\n📊 Statistiques descriptives :")
print(df.describe().round(2).to_string())

# --- Distribution de la variable cible ---
print("\n🎯 Distribution de la variable cible (HiringDecision) :")
counts = df['HiringDecision'].value_counts()
print(f"   0 — Non recruté : {counts[0]}  ({counts[0]/len(df)*100:.1f}%)")
print(f"   1 — Recruté     : {counts[1]}  ({counts[1]/len(df)*100:.1f}%)")
print(f"   ⚠️  Déséquilibre : ratio = {counts[0]/counts[1]:.2f}:1")

# --- Valeurs manquantes ---
print(f"\n🔍 Valeurs manquantes : {df.isnull().sum().sum()} (aucune)")

# ============================================================
# 2. Visualisations
# ============================================================

# --- Graphique 1 : Histogramme de l'expérience par décision ---
fig, ax = plt.subplots(figsize=(10, 6))
couleurs = {0: '#E74C3C', 1: '#2ECC71'}
etiquettes = {0: 'Non recruté', 1: 'Recruté'}

for decision in [0, 1]:
    subset = df[df['HiringDecision'] == decision]
    ax.hist(subset['ExperienceYears'], bins=16, alpha=0.7,
            color=couleurs[decision], label=etiquettes[decision],
            edgecolor='white', linewidth=0.5)

ax.set_title("Distribution des années d'expérience par décision de recrutement",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Années d'expérience", fontsize=12)
ax.set_ylabel("Nombre de candidats", fontsize=12)
ax.legend(title="Décision", fontsize=11, title_fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.savefig("plots/histogramme_experience.png")
plt.close()
print("\n✅ Graphique 1 sauvegardé : plots/histogramme_experience.png")

# --- Graphique 2 : Matrice de corrélation ---
fig, ax = plt.subplots(figsize=(12, 9))
correlation = df.corr()
masque = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(correlation, mask=masque, cmap=cmap, center=0,
            annot=True, fmt='.2f', linewidths=0.8,
            square=True, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Coefficient de corrélation"},
            annot_kws={"size": 9})

ax.set_title("Matrice de corrélation des variables",
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("plots/matrice_correlation.png")
plt.close()
print("✅ Graphique 2 sauvegardé : plots/matrice_correlation.png")

# --- Graphique 3 : Boxplot du score d'entretien vs décision ---
fig, ax = plt.subplots(figsize=(8, 6))

bp = sns.boxplot(x='HiringDecision', y='InterviewScore', data=df,
                 hue='HiringDecision', palette={0: '#E74C3C', 1: '#2ECC71'},
                 legend=False, ax=ax, width=0.5,
                 flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

ax.set_xticklabels(['Non recruté (0)', 'Recruté (1)'], fontsize=12)
ax.set_title("Score d'entretien selon la décision de recrutement",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Décision de recrutement", fontsize=12)
ax.set_ylabel("Score d'entretien", fontsize=12)
ax.spines[['top', 'right']].set_visible(False)

# Ajout des médianes en annotation
medianes = df.groupby('HiringDecision')['InterviewScore'].median()
for i, med in enumerate(medianes):
    ax.annotate(f'Médiane: {med:.0f}', xy=(i, med),
                xytext=(i + 0.25, med + 3), fontsize=10, color='black',
                fontweight='bold')

plt.savefig("plots/boxplot_interview_score.png")
plt.close()
print("✅ Graphique 3 sauvegardé : plots/boxplot_interview_score.png")

# --- Graphique 4 : Taux de recrutement par niveau d'éducation ---
fig, ax = plt.subplots(figsize=(9, 6))

# Labels marocains pour les niveaux d'éducation
labels_education = {1: 'Bac', 2: 'Licence', 3: 'Master', 4: 'Doctorat'}

# Calcul du taux de recrutement par niveau
taux_recrutement = df.groupby('EducationLevel')['HiringDecision'].mean() * 100
taux_recrutement.index = taux_recrutement.index.map(labels_education)

# Palette de couleurs dégradée
couleurs_barres = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

barres = ax.bar(taux_recrutement.index, taux_recrutement.values,
                color=couleurs_barres, edgecolor='white', linewidth=1.5, width=0.6)

# Ajout des pourcentages au-dessus des barres
for barre, val in zip(barres, taux_recrutement.values):
    ax.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 1.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_title("Taux de recrutement par niveau d'éducation",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Niveau d'éducation", fontsize=12)
ax.set_ylabel("Taux de recrutement (%)", fontsize=12)
ax.set_ylim(0, max(taux_recrutement.values) + 12)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(y=df['HiringDecision'].mean() * 100, color='red',
           linestyle='--', alpha=0.6, label=f"Moyenne globale ({df['HiringDecision'].mean()*100:.1f}%)")
ax.legend(fontsize=11)

plt.savefig("plots/taux_recrutement_education.png")
plt.close()
print("✅ Graphique 4 sauvegardé : plots/taux_recrutement_education.png")

print("\n" + "=" * 60)
print("  ✅ EXPLORATION TERMINÉE — 4 graphiques dans plots/")
print("=" * 60)
