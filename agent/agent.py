import pandas as pd
import pickle
import json
import os
from datetime import datetime
from alerter import send_alert

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
DATA_PATH = os.path.join(BASE_DIR, "new_students.csv")
LOG_PATH = os.path.join(BASE_DIR, "agent", "agent_log.txt")

# Charger modele
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
features = json.load(open(FEATURES_PATH, "r"))
config = json.load(open(CONFIG_PATH, "r"))
threshold = config["optimal_threshold"]

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def analyze_student(row):
    data = {feat: 0 for feat in features}
    for col in row.index:
        if col in data:
            data[col] = row[col]
    data["parent_edu_mean"] = (data.get("Medu", 0) + data.get("Fedu", 0)) / 2
    data["total_alcohol"] = data.get("Dalc", 0) + data.get("Walc", 0)
    data["high_risk"] = 1 if (data.get("failures", 0) > 0 and data.get("absences", 0) > 10) else 0
    data["total_support"] = data.get("schoolsup", 0) + data.get("famsup", 0)
    df = pd.DataFrame([data])[features]
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0, 1]
    return proba

def get_risk_factors(row, proba):
    factors = []
    if row.get("failures", 0) > 0:
        factors.append(f"{int(row['failures'])} echec(s) passe(s)")
    if row.get("absences", 0) > 15:
        factors.append(f"{int(row['absences'])} absences")
    if row.get("goout", 0) >= 4:
        factors.append(f"Vie sociale intense ({int(row['goout'])}/5)")
    if row.get("higher", 1) == 0:
        factors.append("Pas de projet d'etudes superieures")
    if (row.get("Dalc", 0) + row.get("Walc", 0)) >= 6:
        factors.append("Consommation d'alcool elevee")
    if not factors:
        factors.append(f"Profil global a risque ({proba:.1%})")
    return factors

def run_agent():
    log("="*50)
    log("AGENT DEMARRE")
    log("="*50)

    if not os.path.exists(DATA_PATH):
        log("ERREUR : Fichier new_students.csv introuvable")
        return

    df = pd.read_csv(DATA_PATH)
    log(f"ANALYSE : {len(df)} eleve(s) a analyser")

    alerts_sent = 0

    for idx, row in df.iterrows():
        student_id = row.get("student_id", f"ELEVE_{idx+1}")

        try:
            proba = analyze_student(row)

            if proba >= threshold:
                log(f"ALERTE : {student_id} - RISQUE DETECTE ({proba:.1%})")
                factors = get_risk_factors(row, proba)
                send_alert(student_id, proba, factors)
                alerts_sent += 1
            else:
                log(f"OK : {student_id} - En securite ({proba:.1%})")

        except Exception as e:
            log(f"ERREUR pour {student_id} : {e}")

    log(f"RESUME : {alerts_sent} alerte(s) sur {len(df)} eleve(s)")
    log("AGENT TERMINE")
    log("="*50)

if __name__ == "__main__":
    run_agent()