# Rapport Final — Prédiction du Décrochage Scolaire

---

## 1. Contexte et Objectif

**Dataset** : 395 étudiants, 33 variables  
**Problème** : Prédire échec (G3 < 10/20)  
**Objectif métier** : Système d'alerte précoce (Recall > 80%)  
**Décision clé** : Exclusion G1/G2 pour détection précoce

---

## 2. Exploration des Données

**Patterns découverts** :
- Abandon progressif (pic G3=0 > G1/G2=0)
- Irréversibilité (élèves faibles début d'année ne rattrapent jamais)
- Variables clés : failures (+0.34), higher (écart 34%), goout (+0.18)

**Feature Engineering** :
- `total_support` : corrélation +0.10
- `parent_edu_mean` : corrélation −0.12
- `high_risk` : corrélation +0.13

---

## 3. Modélisation

**4 modèles testés** :

| Modèle | Recall | Precision | Verdict |
|--------|--------|-----------|---------|
| Logistic Regression | 65.38% | 47.22% | ✅ Baseline solide |
| Random Forest | 42.31% | 58.00% | ❌ Trop conservateur |
| XGBoost | 46.15% | 57.00% | ❌ Idem RF |
| Neural Network | 26.92% | 50.00% | ❌ Dataset trop petit |

**Modèle retenu** : Logistic Regression (meilleur Recall + AUC)

---

## 4. Optimisation du Seuil

**Seuil standard (0.50)** : Recall 65.38%, Precision 47.22%  
**Seuil optimisé (0.399)** : Recall 80.77%, Precision 44.68%

**Trade-off** : +4 décrocheurs détectés, +7 fausses alertes

**Justification** : Coût fausse alerte << Coût décrocheur raté

---

## 5. Interprétabilité (SHAP)

**Top 3 variables** :
1. `failures` (historique échecs)
2. `goout` (vie sociale)
3. `total_support` (fragilité installée) ← Feature Engineering validé

**Interactions détectées** : failures × absences = risque multiplicatif

**Éthique** : Pas de biais démographique (sex, address en bas du classement)

---

## 6. Résultats Finaux

**Performance** :
- Recall 80.77% (objectif atteint ✅)
- Precision 44.68%
- F1-Score 0.578
- AUC 0.734

**Cas d'usage** : Élève #0 (décrocheur silencieux) détecté malgré présence régulière

---

## 7. Limites et Perspectives

**Plafond identifié** : 5 décrocheurs ratés (19%) = profils atypiques hors variables

**Amélioration future** :
- Intégrer données temps réel (retards, sanctions)
- Évaluations formatives courtes
- Notes devoirs (sans attendre G1/G2)

---

## 8. Conclusion

**Objectif atteint** : Recall > 80% sans G1/G2  
**Livrable** : Modèle explicable et actionnable  
**Impact** : Système d'alerte précoce déployable en établissement scolaire

**Next steps** : Application Streamlit + déploiement pilote