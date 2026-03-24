import joblib
import numpy as np
from preprocess import load_ravdess_data
from features   import build_feature_matrix
from train      import prepare_data, train_random_forest, train_neural_network, plot_results

print("=" * 50)
print("SPEECH EMOTION RECOGNITION — FULL PIPELINE")
print("=" * 50)

# ── Step 1: Load data ──────────────────────────────
print("\n[1/5] Loading RAVDESS dataset...")
df = load_ravdess_data('Data_set')

# ── Step 2: Extract features ───────────────────────
print("\n[2/5] Extracting audio features...")
X, y = build_feature_matrix(df)

# ── Step 3: Prepare data ───────────────────────────
print("\n[3/5] Preparing data (encode + scale + split)...")
X_train, X_test, y_train, y_test, scaler, le = prepare_data(X, y)

# Save scaler and label encoder
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le,     'models/label_encoder.pkl')
print("Scaler and label encoder saved to models/")

# ── Step 4: Train models ───────────────────────────
print("\n[4/5] Training models...")
rf_model, y_pred_rf           = train_random_forest(X_train, X_test, y_train, y_test, le)
nn_model, history, y_pred_nn  = train_neural_network(X_train, X_test, y_train, y_test, le)

# ── Step 5: Plot results ───────────────────────────
print("\n[5/5] Saving plots...")
plot_results(history, y_test, y_pred_nn, y_pred_rf, le)

print("\n" + "=" * 50)
print("PIPELINE COMPLETE!")
print("Models saved in : models/")
print("Plots saved in  : outputs/")
print("=" * 50)