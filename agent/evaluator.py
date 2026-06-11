import pandas as pd
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
X_TRAIN_PATH = os.path.join(BASE_DIR, "data", "X_train.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "data", "X_test.csv")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "data", "y_train.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "data", "y_test.csv")
LOG_PATH = os.path.join(BASE_DIR, "agent", "agent_log.txt")

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def evaluate_current_model():
    """Evalue les performances du modele actuel sur X_test."""
    
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    features = json.load(open(FEATURES_PATH, "r"))
    config = json.load(open(CONFIG_PATH, "r"))
    threshold = config["optimal_threshold"]

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    X_test_scaled = scaler.transform(X_test[features])
    probas = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probas >= threshold).astype(int)

    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    log(f"EVALUATION ACTUELLE - Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
    
    return recall, precision, f1

def retrain_if_needed(min_recall=0.75):
    """Reentraine le modele si le recall tombe sous le seuil."""
    
    log("="*50)
    log("EVALUATEUR DEMARRE")
    log("="*50)

    recall, precision, f1 = evaluate_current_model()

    if recall >= min_recall:
        log(f"MODELE OK : Recall {recall:.2%} >= seuil {min_recall:.2%}")
        log("Aucun reentainement necessaire.")
        return False

    log(f"ALERTE MODELE : Recall {recall:.2%} < seuil {min_recall:.2%}")
    log("Reentainement automatique en cours...")

    # Charger donnees
    features = json.load(open(FEATURES_PATH, "r"))
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    # Nouveau scaler
    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(X_train[features])
    X_test_scaled = new_scaler.transform(X_test[features])

    # Nouveau modele
    new_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
    new_model.fit(X_train_scaled, y_train)

    # Evaluer nouveau modele
    config = json.load(open(CONFIG_PATH, "r"))
    threshold = config["optimal_threshold"]
    
    new_probas = new_model.predict_proba(X_test_scaled)[:, 1]
    new_preds = (new_probas >= threshold).astype(int)
    
    new_recall = recall_score(y_test, new_preds)
    new_precision = precision_score(y_test, new_preds)
    new_f1 = f1_score(y_test, new_preds)

    log(f"NOUVEAU MODELE - Recall: {new_recall:.4f} | Precision: {new_precision:.4f} | F1: {new_f1:.4f}")

    # Sauvegarder seulement si amelioration
    if new_recall > recall:
        pickle.dump(new_scaler, open(SCALER_PATH, "wb"))
        pickle.dump(new_model, open(MODEL_PATH, "wb"))
        log(f"MODELE MIS A JOUR : {recall:.2%} -> {new_recall:.2%}")
        return True
    else:
        log("Nouveau modele moins performant. Ancien modele conserve.")
        return False

if __name__ == "__main__":
    retrain_if_needed()