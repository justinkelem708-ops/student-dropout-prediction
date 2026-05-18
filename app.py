import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Configuration
st.set_page_config(page_title="Prédiction Décrochage Scolaire", page_icon="🎓", layout="wide")

# Charger modèle
@st.cache_resource
def load_model():
    model = pickle.load(open('models/logistic_regression_final.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    features = json.load(open('models/feature_names.json', 'r'))
    config = json.load(open('models/config.json', 'r'))
    return model, scaler, features, config

model, scaler, features, config = load_model()

# Titre
st.title("🎓 Système de Prédiction du Décrochage Scolaire")
st.markdown("**Système d'alerte précoce** pour identifier les élèves à risque d'échec avant les premières évaluations.")
st.markdown("**Performance** : Recall 80.77% | Precision 44.68% | F1-Score 0.578")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📊 À propos du modèle")
    st.metric("Recall (Couverture)", "80.77%", help="% de décrocheurs détectés")
    st.metric("Precision", "44.68%", help="Fiabilité des alertes")
    st.metric("Seuil optimal", f"{config['optimal_threshold']:.3f}")
    st.markdown("---")
    st.markdown("""
    **Variables clés** :
    1. Historique d'échecs
    2. Fréquence des sorties
    3. Soutien cumulé (créée ✅)
    
    **Décision stratégique** :  
    Exclusion G1/G2 pour détection précoce
    """)

# Onglets
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📈 Analyse", "ℹ️ Documentation"])

with tab1:
    st.header("Saisir les informations de l'élève")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 Profil Académique")
        school = st.radio("École", ["Gabriel Pereira", "Mousinho da Silveira"], horizontal=True)
        failures = st.number_input("Échecs passés", 0, 4, 0, help="Redoublements antérieurs")
        absences = st.number_input("Absences", 0, 93, 0, help="Absences cumulées")
        studytime = st.selectbox("Temps d'étude/semaine", 
                                 ["Moins de 2h", "2-5h", "5-10h", "Plus de 10h"])
    
    with col2:
        st.subheader("👨‍👩‍👧‍👦 Contexte Familial")
        Medu = st.selectbox("Éducation mère", 
                            ["Aucune", "Primaire", "Collège", "Lycée", "Supérieur"])
        Fedu = st.selectbox("Éducation père", 
                            ["Aucune", "Primaire", "Collège", "Lycée", "Supérieur"])
        famrel = st.slider("Relations familiales", 1, 5, 3, help="1=Très mauvaises, 5=Excellentes")
        famsup = st.radio("Soutien familial", ["Non", "Oui"], horizontal=True)
    
    with col3:
        st.subheader("🎯 Comportement")
        goout = st.slider("Fréquence sorties", 1, 5, 3, help="1=Très rare, 5=Très fréquent")
        Dalc = st.slider("Alcool semaine", 1, 5, 1, help="1=Très faible, 5=Très élevé")
        Walc = st.slider("Alcool week-end", 1, 5, 1, help="1=Très faible, 5=Très élevé")
        higher = st.radio("Études supérieures", ["Oui", "Non"], horizontal=True)
    
    with st.expander("⚙️ Paramètres additionnels (optionnel)"):
        col4, col5 = st.columns(2)
        with col4:
            sex = st.radio("Sexe", ["Féminin", "Masculin"], horizontal=True)
            age = st.number_input("Âge", 15, 22, 17)
            address = st.radio("Adresse", ["Urbain", "Rural"], horizontal=True)
            Pstatus = st.radio("Parents", ["Ensemble", "Séparés"], horizontal=True)
        with col5:
            traveltime = st.selectbox("Trajet école", ["<15min", "15-30min", "30min-1h", ">1h"])
            activities = st.radio("Activités extra-scolaires", ["Non", "Oui"], horizontal=True)
            internet = st.radio("Internet à la maison", ["Oui", "Non"], horizontal=True)
            romantic = st.radio("En couple", ["Non", "Oui"], horizontal=True)
    
    if st.button("🔮 Prédire le risque de décrochage", type="primary", use_container_width=True):
        
        # Conversion
        school_val = 1 if school == "Gabriel Pereira" else 0
        studytime_val = ["Moins de 2h", "2-5h", "5-10h", "Plus de 10h"].index(studytime) + 1
        Medu_val = ["Aucune", "Primaire", "Collège", "Lycée", "Supérieur"].index(Medu)
        Fedu_val = ["Aucune", "Primaire", "Collège", "Lycée", "Supérieur"].index(Fedu)
        sex_val = 1 if sex == "Féminin" else 0
        address_val = 1 if address == "Urbain" else 0
        Pstatus_val = 1 if Pstatus == "Ensemble" else 0
        traveltime_val = ["<15min", "15-30min", "30min-1h", ">1h"].index(traveltime) + 1
        famsup_val = 1 if famsup == "Oui" else 0
        
        # Features créées
        parent_edu_mean_val = (Medu_val + Fedu_val) / 2
        total_alcohol_val = Dalc + Walc
        high_risk_val = 1 if (failures > 0 and absences > 10) else 0
        total_support_val = 0 + famsup_val  # schoolsup=0 par défaut
        
        # ORDRE EXACT selon feature_names.json
        data = {
            'school': school_val,
            'sex': sex_val,
            'age': age,
            'address': address_val,
            'famsize': 1,
            'Pstatus': Pstatus_val,
            'Medu': Medu_val,
            'Fedu': Fedu_val,
            'Mjob': 0,
            'Fjob': 0,
            'reason': 0,
            'guardian': 0,
            'traveltime': traveltime_val,
            'studytime': studytime_val,
            'failures': failures,
            'schoolsup': 0,
            'famsup': famsup_val,
            'paid': 0,
            'activities': 1 if activities == "Oui" else 0,
            'nursery': 1,
            'higher': 1 if higher == "Oui" else 0,
            'internet': 1 if internet == "Oui" else 0,
            'romantic': 1 if romantic == "Oui" else 0,
            'famrel': famrel,
            'freetime': 3,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': 3,
            'absences': absences,
            'parent_edu_mean': parent_edu_mean_val,
            'total_alcohol': total_alcohol_val,
            'high_risk': high_risk_val,
            'total_support': total_support_val
        }
        
        # DataFrame avec colonnes dans l'ordre EXACT
        df = pd.DataFrame([data])[features]
        
        # Prédire
        df_scaled = scaler.transform(df)
        proba = model.predict_proba(df_scaled)[0, 1]
        pred = 1 if proba >= config['optimal_threshold'] else 0
        
        # Affichage
        st.divider()
        st.header("📊 Résultats de la Prédiction")
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        
        with col_b:
            if pred == 1:
                st.error("⚠️ **ALERTE : Risque de décrochage détecté**")
            else:
                st.success("✅ **Pas d'alerte : Élève en sécurité**")
            
            st.metric("Probabilité de décrochage", f"{proba:.1%}")
            st.progress(proba)
            
            st.markdown("---")
            st.subheader("💡 Interprétation")
            
            if pred == 1:
                st.markdown(f"""
                **Niveau de risque** : {'Critique' if proba > 0.7 else 'Élevé' if proba > 0.5 else 'Modéré'}
                
                **Facteurs de risque identifiés** :
                """)
                
                if failures > 0:
                    st.markdown(f"• **{failures} échec(s) passé(s)** — Signal dominant")
                if absences > 15:
                    st.markdown(f"• **{absences} absences** — Décrochage progressif")
                if goout >= 4:
                    st.markdown(f"• **Vie sociale intense** ({goout}/5)")
                if higher == "Non":
                    st.markdown("• **Pas de projet d'études supérieures**")
                if total_alcohol_val >= 6:
                    st.markdown(f"• **Consommation d'alcool élevée** ({total_alcohol_val}/10)")
                
                st.markdown("""
                **Recommandations** :
                1. Entretien individuel
                2. Tutorat académique ciblé
                3. Suivi assiduité
                4. Orientation scolaire
                """)
            else:
                st.markdown("Profil **stable**. Aucune intervention urgente.")
                st.markdown("**Facteurs protecteurs** :")
                if failures == 0:
                    st.markdown("• Pas d'échec passé")
                if absences < 10:
                    st.markdown("• Bonne assiduité")
                if higher == "Oui":
                    st.markdown("• Projet d'études clair")

with tab2:
    st.header("📈 Analyse du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variables importantes (SHAP)")
        st.bar_chart(pd.DataFrame({
            'Variable': ['failures', 'goout', 'total_support', 'age', 'reason'],
            'Impact': [0.15, 0.12, 0.10, 0.08, 0.07]
        }).set_index('Variable'))
        st.markdown("""
        **Top 3** :
        1. **failures** : Historique d'échecs
        2. **goout** : Fréquence sorties
        3. **total_support** : Soutien (créée ✅)
        """)
    
    with col2:
        st.subheader("Performance")
        st.markdown(f"""
        | Métrique | Valeur |
        |----------|--------|
        | Recall | **80.77%** ✅ |
        | Precision | 44.68% |
        | F1-Score | 0.578 |
        | AUC-ROC | 0.734 |
        | Seuil | {config['optimal_threshold']:.3f} |
        
        **Arbitrage** :  
        Maximiser Recall au prix de fausses alertes.
        
        Coût décrocheur raté >> coût fausse alerte
        """)

with tab3:
    st.header("ℹ️ Documentation")
    st.markdown("""
    ## À propos
    
    **Dataset** : 395 étudiants, 33 variables  
    **Problème** : Classification binaire (échec < 10/20)  
    **Modèle** : Régression Logistique + threshold optimisé
    
    ## Décision stratégique
    
    **Exclusion G1/G2** pour détection précoce avant le 1er examen.
    
    Le modèle utilise :
    - Historique scolaire (échecs)
    - Comportement (absences, sorties, alcool)
    - Contexte familial (éducation, soutien)
    
    ## Limites
    
    - 19% décrocheurs non détectés (profils atypiques)
    - 55% fausses alertes (trade-off assumé)
    - Calibré sur données portugaises (2008)
    
    ## Source
    
    Cortez & Silva (2008). *Using Data Mining to Predict Student Performance*.  
    UCI Machine Learning Repository.
    
    ## Code
    
    [GitHub - student-dropout-prediction](https://github.com/justinkelem708-ops/student-dropout-prediction)
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Développé par Justin | Projet Master AI/DS 2027
</div>
""", unsafe_allow_html=True)