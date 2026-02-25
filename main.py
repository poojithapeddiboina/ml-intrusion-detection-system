from autoencoder_model import build_autoencoder, train_autoencoder, predict_autoencoder
from data_preprocessing import load_data, clean_data, split_data_zero_day, scale_data
from evaluation import evaluate_model
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import shap
import numpy as np

# ---------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------
print("Step 1: Loading Data")
df = load_data("data.csv")

# ---------------------------------------------------
# 2️⃣ Clean Data
# ---------------------------------------------------
print("Step 2: Cleaning Data")
df = clean_data(df)

print("Available Attack Types:")
print(df["AttackType"].unique())

# ---------------------------------------------------
# 3️⃣ Zero-Day Setup
# ---------------------------------------------------
print("Step 3: Zero-Day Split")
zero_day_attack = "DDoS"   # Change if needed

X_train, X_train_normal, X_test, y_train, y_test = split_data_zero_day(df, zero_day_attack)

print("Training Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)

# ---------------------------------------------------
# 4️⃣ Scaling
# ---------------------------------------------------
print("Step 4: Scaling")
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
X_train_normal_scaled = scaler.transform(X_train_normal)

# ---------------------------------------------------
# 5️⃣ Build Autoencoder
# ---------------------------------------------------
print("Step 5: Building Autoencoder")
input_dim = X_train_normal_scaled.shape[1]
model = build_autoencoder(input_dim)

# ---------------------------------------------------
# 6️⃣ Train Autoencoder
# ---------------------------------------------------
print("Step 6: Training Autoencoder")
train_autoencoder(model, X_train_normal_scaled)

# ---------------------------------------------------
# 7️⃣ Autoencoder Prediction & Evaluation
# ---------------------------------------------------
print("Step 7: Autoencoder Predicting")
predictions, scores = predict_autoencoder(model, X_test_scaled, X_train_normal_scaled)

print("Step 8: Autoencoder Evaluation")
evaluate_model(y_true=y_test, y_pred=predictions, y_scores=scores)

# Zero-Day Recall for Autoencoder
zero_day_mask = (y_test == 1)
if np.sum(zero_day_mask) > 0:
    zero_day_recall = np.sum(predictions[zero_day_mask] == 1) / np.sum(zero_day_mask)
    print("Zero-Day Recall (Autoencoder):", zero_day_recall)
else:
    print("No Zero-Day samples found for Autoencoder.")

# ---------------------------------------------------
# 8️⃣ Isolation Forest
# ---------------------------------------------------
print("\nStep 9: Isolation Forest Training")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_normal_scaled)

# Predict labels
iso_pred = iso_forest.predict(X_test_scaled)
iso_pred_binary = np.where(iso_pred == 1, 0, 1)

# Get anomaly scores for ROC-AUC
iso_scores = -iso_forest.decision_function(X_test_scaled)

print("Evaluating Isolation Forest...")
evaluate_model(y_true=y_test, y_pred=iso_pred_binary, y_scores=iso_scores)

# Zero-Day Recall for Isolation Forest
if np.sum(zero_day_mask) > 0:
    zero_day_recall_iso = np.sum(iso_pred_binary[zero_day_mask] == 1) / np.sum(zero_day_mask)
    print("Zero-Day Recall (Isolation Forest):", zero_day_recall_iso)

# ---------------------------------------------------
# 9️⃣ Random Forest
# ---------------------------------------------------
print("\nStep 10: Random Forest Training")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)

# rf.predict_proba might return shape (n_samples, n_classes) — handle safe indexing
if rf_pred.ndim == 1 or rf.predict_proba(X_test_scaled).shape[1] == 1:
    rf_pred_prob = rf_pred  # fallback for single-class probability
else:
    rf_pred_prob = rf.predict_proba(X_test_scaled)[:,1]

print("Evaluating Random Forest...")
evaluate_model(y_true=y_test, y_pred=rf_pred, y_scores=rf_pred_prob)

# Zero-Day Recall for Random Forest
if np.sum(zero_day_mask) > 0:
    zero_day_recall_rf = np.sum(rf_pred[zero_day_mask] == 1) / np.sum(zero_day_mask)
    print("Zero-Day Recall (Random Forest):", zero_day_recall_rf)

# ---------------------------------------------------
# 🔟 SHAP Explainability for Random Forest
# ---------------------------------------------------
print("\nStep 11: Random Forest Explainability with SHAP")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)

# Ensure 2D shap_values for summary_plot
if isinstance(shap_values, list):
    shap_class1 = shap_values[1]  # multi-class RF -> pick class 1
else:
    shap_class1 = np.array(shap_values)
    if shap_class1.ndim == 1:
        shap_class1 = shap_class1.reshape(-1, 1)

# Sample first 5000 rows for speed
sample_idx = np.arange(min(5000, shap_class1.shape[0]))
shap.summary_plot(shap_class1[sample_idx], X_test_scaled[sample_idx], feature_names=X_train.columns)

print("\nDONE ✅ All models evaluated and explainability done.")


# ---------------------------------------------------
# 🔹 Save All Models
# ---------------------------------------------------
import joblib
from tensorflow.keras.models import save_model

# Autoencoder
save_model(model, "autoencoder_model.h5")

# Isolation Forest
joblib.dump(iso_forest, "isolation_forest.pkl")

# Random Forest
joblib.dump(rf, "random_forest.pkl")

# Scaler
joblib.dump(scaler, "scaler.pkl")

print("✅ All models and scaler saved!")
