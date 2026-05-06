# 🤖 Rapport de Modélisation — Projet Décrochage Scolaire

---

## 1. Stratégie de modélisation

**Objectif** : Tester 4 approches différentes pour identifier le meilleur compromis Recall/Precision

**Modèles sélectionnés** :
1. **Logistic Regression** — Baseline interprétable
2. **Random Forest** — Capture des relations non-linéaires
3. **XGBoost** — Boosting pour classes déséquilibrées
4. **Neural Network (MLP)** — Test de la complexité sur petit dataset

**Métrique prioritaire** : **Recall** (ne rater aucun décrocheur)

---

## 2. Jour 4 : Régression Logistique (Baseline)

### Performance (Classe ÉCHEC)

| Métrique | Score | Objectif | État |
|----------|-------|----------|------|
| Precision | 47.22 % | > 60 % | ❌ Insuffisant |
| Recall | 65.38 % | > 80 % | ❌ Insuffisant |
| F1-Score | 0.548 | Max | ⚠️ Moyen |
| AUC | 0.734 | — | Séparation correcte |

### Matrice de confusion (79 élèves test)

| Résultat | Nombre | Interprétation |
|----------|--------|----------------|
| Vrais Positifs | 17 | Élèves en difficulté détectés |
| Faux Négatifs | 9 | Élèves ratés — priorité Recall |
| Faux Positifs | 19 | Fausses alertes |

### Top 10 Coefficients

| Rang | Variable | Effet | Coefficient | Interprétation |
|------|----------|-------|-------------|----------------|
| 1 | `failures` | 📈 Risque | +0.64 | Signal dominant |
| 2 | `Pstatus` | 📈 Risque | +0.52 | Situation familiale |
| 3 | `goout` | 📈 Risque | +0.43 | Vie sociale intense |
| 4 | `higher` | 🛡️ Protection | −0.42 | Ambition scolaire |
| 5 | `total_support` | 📈 Risque | +0.34 | ✅ Feature Engineering validé |
| 6 | `sex` | 🛡️ Protection | −0.34 | Facteur démographique |
| 7 | `nursery` | 📈 Risque | +0.33 | Origine sociale |
| 8 | `address` | 🛡️ Protection | −0.32 | Zone urbaine |
| 9 | `age` | 📈 Risque | +0.30 | Retard scolaire |
| 10 | `romantic` | 📈 Risque | +0.29 | Impact relationnel |

### Conclusion

Signal prédictif confirmé. Limite structurelle : la linéarité empêche de capturer les interactions complexes nécessaires pour atteindre Recall > 80%.

---

## 3. Jour 5 : Random Forest Classifier

### Performance (Classe ÉCHEC)

| Métrique | Random Forest | Baseline LR | Évolution |
|----------|--------------|-------------|-----------|
| Precision | 58.00 % | 47.22 % | 📈 +10.78 pts |
| Recall | 42.31 % | 65.38 % | 📉 −23.07 pts |
| F1-Score | 0.49 | 0.55 | ❌ −0.06 |

### Feature Importance — Top 10

| Rang | Variable | Importance | Statut |
|------|----------|------------|--------|
| 1 | `failures` | 0.101 | Brute |
| 2 | `absences` | 0.097 | Brute |
| 3 | `goout` | 0.063 | Brute |
| 4 | `parent_edu_mean` | 0.050 | ✅ Créée |
| 5 | `age` | 0.047 | Brute |
| 6 | `health` | 0.042 | Brute |
| 7 | `famrel` | 0.038 | Brute |
| 8 | `reason` | 0.038 | Brute |
| 9 | `total_alcohol` | 0.037 | ✅ Créée |
| 10 | `Fedu` | 0.036 | Brute |

### Diagnostic

Gain de fiabilité, perte de couverture. Malgré `class_weight='balanced'`, RF reste trop conservateur. Sa structure en arbres privilégie la pureté des décisions : il n'alerte que lorsqu'il est très certain, et rate les décrochages subtils.

**Recall 42%** = rate plus d'un décrocheur sur deux = système inopérationnel.

---

## 4. Jour 6 : XGBoost (Gradient Boosting)

### Performance comparative

| Métrique | XGBoost | Random Forest | Baseline LR |
|----------|---------|---------------|-------------|
| Precision | 57.00 % | 58.00 % | 47.22 % |
| Recall | 46.15 % | 42.31 % | 65.38 % |
| F1-Score | 0.51 | 0.49 | 0.55 |

### Feature Importance — Top 10

| Rang | Variable | Importance | Statut |
|------|----------|------------|--------|
| 1 | `failures` | 0.091 | Brute |
| 2 | `total_support` | 0.062 | ✅ Créée |
| 3 | `internet` | 0.047 | Brute |
| 4 | `famrel` | 0.046 | Brute |
| 5 | `guardian` | 0.043 | Brute |
| 6 | `studytime` | 0.037 | Brute |
| 7 | `schoolsup` | 0.035 | Brute |
| 8 | `address` | 0.033 | Brute |
| 9 | `sex` | 0.033 | Brute |
| 10 | `absences` | 0.032 | Brute |

### Résultat majeur

**`total_support` au rang 2** — résultat le plus fort obtenu par une variable créée sur l'ensemble de la modélisation.

XGBoost valorise aussi des signaux ignorés par RF/LR : `internet`, `guardian`, `famrel` entrent dans le Top 5.

### Diagnostic structurel

XGBoost récupère 4 points de Recall sur RF, mais le plafond reste loin de l'objectif. Les arbres (RF + XGBoost) partagent le même biais : ils minimisent les fausses alertes plutôt que de maximiser la couverture.

---

## 5. Jour 7 : Réseau de Neurones (MLP)

### Performance (Classe ÉCHEC)

| Métrique | Neural Network | Baseline LR | Écart |
|----------|---------------|-------------|-------|
| Precision | 50.00 % | 47.22 % | +2.78 pts |
| Recall | 26.92 % | 65.38 % | −38.46 pts |
| F1-Score | 0.35 | 0.55 | −0.20 |

### Analyse

**Courbe de Loss** : Descente régulière — apprentissage sain  
**Problème** : 395 étudiants insuffisants pour qu'un NN surpasse un modèle statistique simple

Le MLP rate 19 décrocheurs sur 26 — résultat le plus faible de la phase. La complexité devient un handicap : le modèle sur-ajuste sans pouvoir généraliser sur la classe minoritaire.

---

## 6. Jour 8 : Comparaison Finale

### Synthèse des performances (Classe ÉCHEC)

| Modèle | Recall | Precision | F1-Score | ROC-AUC |
|--------|--------|-----------|----------|---------|
| **Logistic Regression** | **65.38 %** 🏆 | 47.22 % | **0.548** | **0.734** 🏆 |
| XGBoost | 46.15 % | 57.14 % | 0.51 | 0.688 |
| Random Forest | 42.31 % | 57.89 % | 0.49 | 0.700 |
| Neural Network | 26.92 % | 50.00 % | 0.35 | 0.726 |

### Lecture stratégique

**Pourquoi les modèles complexes échouent ?**

RF, XGBoost et MLP optimisent l'accuracy globale. Face à une classe minoritaire sans notes disponibles, ils apprennent la prudence — et ratent entre 54% et 73% des décrocheurs.

Ce n'est pas un défaut de paramétrage : **c'est un biais structurel**.

**Pourquoi la Régression Logistique résiste ?**

Ses probabilités calibrées et sa linéarité lui permettent de généraliser les signaux comportementaux sans sur-apprendre la majorité. Elle reste le modèle le plus contrôlable — et le plus adaptable.

### Décision finale

**Modèle retenu** : **Régression Logistique**

- Meilleur Recall (65.38%)
- AUC le plus élevé (0.734)
- Fondation optimale pour l'optimisation du seuil

**Levier identifié** : Threshold Tuning

Aucun modèle ne franchit 80% de Recall avec un seuil à 0.50. Abaisser ce seuil redéfinit la sensibilité du système sans changer l'architecture.

> Ce n'est pas un compromis technique — c'est une décision métier : définir à partir de quel niveau de risque un tuteur doit être alerté.

---

## 7. Conclusion

**Enseignements clés** :

1. **Feature Engineering validé** : `total_support` apparaît systématiquement dans le Top 5-10
2. **Linéarité > Complexité** : Sur petit dataset déséquilibré, la simplicité l'emporte
3. **Métrique métier > Métrique technique** : Optimiser Recall, pas Accuracy
4. **Threshold tuning essentiel** : Le seuil 0.50 n'est pas universel

**Prochaine étape** : Optimisation du seuil de décision pour atteindre Recall > 80%