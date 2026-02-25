# app.py
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Intrusion Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# 1️⃣ Load Models
# ---------------------------------------------------

print("Loading models...")

# Autoencoder (.h5) — use compile=False to avoid deserialization issues
autoencoder = load_model("autoencoder_model.h5", compile=False)

# Isolation Forest (.pkl)
iso_forest = joblib.load("isolation_forest.pkl")

# Random Forest (.pkl)
rf = joblib.load("random_forest.pkl")

print("Models loaded successfully ✅")

# ---------------------------------------------------
# 2️⃣ Define Prediction Helper Functions
# ---------------------------------------------------

def preprocess_input(df, scaler=None):
    """Scale input data for models"""
    numeric_df = df.select_dtypes(include=[np.number])
    if scaler:
        X_scaled = scaler.transform(numeric_df)
    else:
        X_scaled = numeric_df.values
    return X_scaled

def predict_autoencoder_model(X_scaled, X_train_normal_scaled):
    """Autoencoder predictions"""
    from autoencoder_model import predict_autoencoder
    pred_labels, pred_scores = predict_autoencoder(autoencoder, X_scaled, X_train_normal_scaled)
    return pred_labels, pred_scores

def predict_isolation(X_scaled):
    pred = iso_forest.predict(X_scaled)
    pred_binary = np.where(pred == 1, 0, 1)
    scores = -iso_forest.decision_function(X_scaled)
    return pred_binary, scores


def predict_random_forest(X_scaled):
    pred = rf.predict(X_scaled)

    # Check number of classes
    if len(rf.classes_) == 2:
        pred_prob = rf.predict_proba(X_scaled)[:, 1]
    else:
        # If only one class trained
        pred_prob = rf.predict_proba(X_scaled)[:, 0]

    return pred, pred_prob

# ---------------------------------------------------
# 3️⃣ API Endpoint
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Save numeric columns only for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X_input = df[numeric_cols].values
    
    # Scale for Autoencoder & Random Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)  # use same scaler as training ideally
    
    # Predictions
    ae_labels, ae_scores = predict_autoencoder_model(X_scaled, X_scaled)
    iso_labels, iso_scores = predict_isolation(X_scaled)
    rf_labels, rf_scores = predict_random_forest(X_scaled)
    
    # Return results
    return {
    "total_rows": len(ae_labels),
    "autoencoder_anomalies": int(np.sum(ae_labels)),
    "isolation_anomalies": int(np.sum(iso_labels)),
    "rf_anomalies": int(np.sum(rf_labels))
}
