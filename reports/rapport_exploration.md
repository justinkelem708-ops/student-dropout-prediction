# Rapport d'Exploration — Projet Décrochage Scolaire
- Objectif = détection précoce, pas prédiction tardive
- Force le modèle à apprendre des facteurs comportementaux/sociaux

**Trade-off assumé** : Moins de précision, mais utilité réelle

---

## 3. Variables clés identifiées

### Top 5 prédicteurs

| Variable | Type | Corrélation/Écart | Interprétation |
|----------|------|-------------------|----------------|
| `failures` | Num | +0.34 | Échecs passés = signal #1 |
| `higher` | Cat | Écart 34% | Ambition scolaire déterminante |
| `goout` | Num | +0.18 | Vie sociale intense = risque |
| `age` | Num | +0.18 | Redoublements = fragilité |
| `schoolsup` | Cat | Écart 14% | Indicateur de fragilité (biais de sélection) |

### Feature Engineering

| Variable créée | Formule | Impact |
|----------------|---------|--------|
| `high_risk` | (failures >0) & (absences >10) | Corrélation +0.13 |
| `total_support` | schoolsup + famsup | Corrélation +0.10 |
| `parent_edu_mean` | (Medu + Fedu) / 2 | Corrélation −0.12 |
| `total_alcohol` | Walc + Dalc | Faible impact |

---

## 4. Patterns découverts

**Abandon progressif** : Pic à G3=0 > G1/G2=0 → décrochage en cours d'année

**Irréversibilité** : Aucun élève avec G1/G2≈0 ne rattrape en G3

**Biais de sélection** : `schoolsup=yes` corrélé avec échec car attribué aux fragiles

**Effet de seuil** : Éducation parents niveau 4 = protection forte

---

## 5. Preprocessing appliqué

- **Encodage** : Variables binaires (yes/no → 1/0), nominales (LabelEncoder)
- **Split** : 80% train / 20% test (stratifié)
- **Normalisation** : StandardScaler sur variables numériques
- **Pas de valeurs manquantes**

**Shapes finales** :
- X_train: (316, 34)
- X_test: (79, 34)

---

## 6. Prochaines étapes

**Jour 4-8** : Modélisation (Logistic Regression, Random Forest, XGBoost, Neural Network)

**Métriques cibles** :
- Recall > 80%
- Precision > 60%
- F1-score maximal

**Livrable final** : App Streamlit + rapport SHAP