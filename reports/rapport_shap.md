# 🔍 Rapport SHAP — Interprétabilité du Modèle

---

## 1. Objectif de l'analyse SHAP

Les coefficients de la Régression Logistique donnent l'importance globale des variables. SHAP (SHapley Additive exPlanations) permet d'expliquer :

- **L'importance relative** de chaque variable (globalement)
- **La contribution exacte** de chaque variable pour chaque élève (localement)

SHAP transforme une prédiction en décision actionnable.

---

## 2. Importance globale des variables

### Summary Plot — Vue d'ensemble

**Axe vertical** : Variables par ordre d'importance  
**Axe horizontal** : Impact SHAP (+ = augmente risque, - = réduit risque)  
**Couleur** : Valeur de la variable (rouge = élevé, bleu = faible)

### Hiérarchie SHAP

| Rang | Variable | Impact | Statut |
|------|----------|--------|--------|
| 1 | `failures` | Élevé | Brute |
| 2 | `goout` | Élevé | Brute |
| 3 | `total_support` | Moyen-Haut | ✅ Créée |
| 4 | `reason` | Moyen | Brute |
| 5 | `age` | Moyen | Brute |

### Convergence avec les coefficients

Le classement SHAP confirme la hiérarchie identifiée par les coefficients de la Baseline.

**`total_support` au rang 3** — au-dessus de l'âge, du sexe et de l'adresse.  
✅ **Feature Engineering validé empiriquement**

---

## 3. Moteurs du risque (Points rouges à droite)

### `failures` — Signal dominant

Les valeurs élevées sont projetées très loin vers la droite. L'historique d'échecs est le déclencheur le plus puissant du modèle.

### `goout` — Comportement social

Une vie sociale intense pousse systématiquement la prédiction vers la zone de risque.

### `total_support` — Rang 3 ✅

Variable créée — points rouges à droite. Le cumul des aides est perçu comme un indicateur de fragilité critique.

### `age` — Retard scolaire

Un âge élevé amplifie la probabilité d'échec, cohérent avec son rôle de proxy du redoublement.

---

## 4. Facteurs de protection (Points à gauche)

### `sex` et `reason`

Amas de points à gauche selon la catégorie — ces variables tirent certains profils vers la réussite.

### `Dalc` — Alcool en semaine

Les faibles consommations (points bleus) se regroupent à gauche. La sobriété en semaine agit comme un facteur de stabilité.

---

## 5. Étude de cas — Élève #0 (Vrai Positif)

### Diagnostic

| Paramètre | Valeur |
|-----------|--------|
| Situation réelle | ÉCHEC |
| Prédiction | ÉCHEC ✅ |
| Probabilité calculée | 58.10 % |
| Seuil d'alerte | 39.9 % |

### Cascade SHAP (Waterfall Plot)

**Point de départ** : Risque moyen global (−0.154)  
**Point d'arrivée** : Score individuel (+0.327)

**Facteurs de risque (poussent vers l'échec)** :

| Variable | Contribution | Interprétation |
|----------|-------------|----------------|
| `failures` | +0.60 | Historique d'échecs — facteur le plus lourd |
| `age` | +0.57 | Retard scolaire — aggrave le profil |
| `address` + `internet` | — | Zone rurale sans accès internet |

**Facteurs protecteurs (tirent vers la réussite)** :

| Variable | Contribution | Interprétation |
|----------|-------------|----------------|
| `total_support` | −0.48 | Faible soutien déclaré |
| `goout` | −0.44 | Vie sociale calme — bouclier principal |
| `absences` | −0.15 | Présence régulière — atténue sans compenser |

### Portrait métier

**Profil : Décrocheur silencieux**

Présent en cours, discret, pas de signaux comportementaux évidents. Mais son passé scolaire et son isolement social construisent un risque structurel que l'observation directe ne détecte pas.

> Sans ce système, cet élève passerait inaperçu jusqu'à l'échec. Avec l'alerte, le tuteur peut cibler une remise à niveau académique — pas un suivi disciplinaire.

---

## 6. Interactions critiques (Dependence Plots)

### Effet multiplicateur de `failures`

L'impact des échecs passés n'est pas linéaire. Dès que `failures` passe de 0 à 1, le score de risque fait un saut brutal.

Avoir échoué une seule fois restructure entièrement le profil de risque — ce n'est pas un facteur graduel, c'est un déclencheur.

### Synergie `absences` × `failures`

Les points rouges (élèves avec échecs passés) se situent systématiquement plus haut sur l'axe vertical, pour un même niveau d'absences.

> Deux élèves avec 15 absences n'ont pas le même risque si l'un a déjà échoué et l'autre non. Le contexte académique amplifie l'impact du comportement.

### Effet de seuil critique

Le cumul "Absences + Échecs" crée une zone de rupture : au-delà d'un certain seuil combiné, le risque de décrochage s'emballe de façon non proportionnelle.

> Le décrochage n'est pas la somme de facteurs isolés — c'est leur interaction qui crée la vulnérabilité.

---

## 7. Vérification de biais

### Variables démographiques analysées

`sex`, `address`, `internet` → Bas classement SHAP

### Conclusion

Pas de biais démographique détecté. Le modèle ne discrimine pas sur des critères sociaux fixes. Il priorise les comportements (`goout`, `absences`) et le parcours (`failures`).

**Le système est éthiquement robuste.**

---

## 8. Valeur ajoutée de SHAP

| Apport | Concrètement |
|--------|-------------|
| **Transparence** | Chaque alerte justifiée par des faits précis |
| **Précision** | Le tuteur sait s'il doit agir sur le social ou l'académique |
| **Confiance** | 81% Recall + logique explicable = adhésion utilisateurs |

> SHAP transforme une prédiction en décision. C'est ce qui fait la différence entre un modèle de laboratoire et un outil utilisable sur le terrain.

---

## 9. Conclusion

**Le modèle raisonne en trois niveaux** :

1. **Scolaire** — `failures` : signal dominant
2. **Social** — `goout`, `age` : amplificateurs de risque
3. **Environnemental** — `total_support` : marqueur de fragilité installée

> SHAP confirme que le modèle raisonne comme un tuteur expérimenté : il regarde d'abord le passé scolaire, puis le comportement, puis le contexte de soutien — dans cet ordre.

**Le système est prêt pour la production.**