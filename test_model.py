import pickle
import pandas as pd
import json

# Charger
model = pickle.load(open('models/logistic_regression_final.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
features = json.load(open('models/feature_names.json', 'r'))

print("="*60)
print("DIAGNOSTIC MODÈLE")
print("="*60)

print(f"\nNombre de features attendues : {len(features)}")
print(f"Features : {features}")

# Créer des données test
test_data = {feat: 0 for feat in features}
test_data['parent_edu_mean'] = 2.5
test_data['age'] = 17

df = pd.DataFrame([test_data])

print(f"\nColonnes DataFrame : {df.columns.tolist()}")
print(f"Shape DataFrame : {df.shape}")

# Test scaler
try:
    df_scaled = scaler.transform(df)
    print("\n✅ Scaler fonctionne")
    print(f"Shape après scaling : {df_scaled.shape}")
except Exception as e:
    print(f"\n❌ ERREUR SCALER : {e}")

# Test model
try:
    proba = model.predict_proba(df_scaled)
    print(f"\n✅ Modèle fonctionne")
    print(f"Probabilité : {proba[0][1]:.2%}")
except Exception as e:
    print(f"\n❌ ERREUR MODÈLE : {e}")