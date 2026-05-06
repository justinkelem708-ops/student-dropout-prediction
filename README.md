# 🎓 Prédiction du Décrochage Scolaire

Système d'alerte précoce pour identifier les élèves à risque d'échec sans attendre les premières évaluations.

---

## 📊 Vue d'ensemble

**Objectif** : Prédire l'échec scolaire (note < 10/20) à partir de variables socio-comportementales

**Performance** :
- ✅ Recall : 80.77% (objectif atteint)
- Precision : 44.68%
- F1-Score : 0.578
- AUC-ROC : 0.734

**Modèle retenu** : Régression Logistique (seuil optimisé à 0.399)

---
## 📁 Structure du Projet

```text
Project1-Dropout/
├── data/           # Données brutes et préprocessées
├── notebooks/      # Jupyter notebooks d'analyse
├── outputs/        # Visualisations et graphiques
├── reports/        # Rapports markdown
├── models/         # Modèles sauvegardés
└── README.md       # Ce fichier de présentation
```

## 🚀 Installation

```bash
# Créer l'environnement
conda create -n ml_env python=3.11 -y
conda activate ml_env

# Installer les dépendances
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

---

## 📈 Résultats Clés

### Variables les plus importantes (SHAP)
1. `failures` : Historique d'échecs passés
2. `goout` : Fréquence des sorties sociales
3. `total_support` : Cumul des soutiens (feature créée)

### Feature Engineering
- `total_support` = schoolsup + famsup
- `parent_edu_mean` = (Medu + Fedu) / 2
- `high_risk` = (failures > 0) & (absences > 10)

### Décision stratégique
**Exclusion de G1 et G2** (notes intermédiaires) pour permettre une détection précoce avant le premier examen.

---

## 📊 Visualisations

Toutes les visualisations sont disponibles dans `outputs/` :
- Courbes Precision-Recall
- Matrices de confusion
- Analyse SHAP complète
- Graphiques d'importance des variables

---

## 🎯 Utilisation

```python
import pickle
import pandas as pd

# Charger le modèle
with open('models/logistic_regression_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger le scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prédire pour un nouvel élève
new_student = pd.DataFrame({...})  # Données élève
new_student_scaled = scaler.transform(new_student)
probability = model.predict_proba(new_student_scaled)[:, 1]

# Alerte si probabilité > seuil optimal
if probability[0] > 0.399:
    print(f"⚠️ ALERTE : Probabilité d'échec = {probability[0]:.2%}")
```

---

## 📝 Rapports Détaillés

- `reports/rapport_exploration.md` : Analyse exploratoire complète
- `reports/rapport_modelisation.md` : Comparaison des 4 modèles
- `reports/rapport_shap.md` : Interprétabilité SHAP
- `reports/rapport_final.md` : Synthèse exécutive

---

## 🔮 Perspectives d'Amélioration

1. Intégrer données temps réel (retards, sanctions)
2. Ajouter évaluations formatives courtes
3. Déployer application Streamlit
4. Tester sur d'autres établissements

---

## 👤 Auteur

**Justin**  
Master Application 2027 - AI/Data Science  
Université Chinoise

---

## 📄 Licence

Dataset : [UCI ML Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)