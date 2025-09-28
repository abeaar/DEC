import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance
import xgboost as xgb 
import matplotlib.pyplot as plt
import seaborn as sns 

# ======================
# Config
# ======================
splits_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits"  
train_csv = os.path.join(splits_root, "train_latent.csv")
val_csv   = os.path.join(splits_root, "val_latent.csv")
test_csv  = os.path.join(splits_root, "test_latent.csv")

# ======================
# Load Dataset
# ======================
from sklearn.preprocessing import LabelEncoder

def load_data(csv_path, label_encoder=None):
    # Baca CSV
    df = pd.read_csv(csv_path)

    # Pilih kolom fitur (semua kolom mulai dengan 'f')
    feature_columns = [col for col in df.columns if col.startswith("f")]
    X = df[feature_columns].to_numpy(dtype=float)

    # Ambil label asli (string)
    y = df["label"].to_numpy()

    if label_encoder is None:
        # Fit LabelEncoder hanya pada label yang benar-benar muncul di dataset ini
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"Fitted LabelEncoder dengan kelas: {list(le.classes_)}")
        return X, y_encoded, le
    else:
        # Gunakan encoder yang sudah ada
        y_encoded = label_encoder.transform(y)
        return X, y_encoded, label_encoder



X_train, y_train, le = load_data(train_csv)
X_val, y_val, _ = load_data(val_csv, le)
X_test, y_test, _ = load_data(test_csv, le)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes: {list(le.classes_)}")

# ======================
# XGBoost Training
# ======================
params = {
    "objective": "multi:softmax",   # klasifikasi multi kelas
    "num_class": len(le.classes_),  # jumlah kelas
    "eval_metric": "mlogloss",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost...")
model = xgb.train(params, dtrain, num_boost_round=200, evals=watchlist, early_stopping_rounds=20)

# ======================
# Evaluation
# ======================
y_pred = model.predict(dtest)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (XGBoost on Latent Features)")
plt.tight_layout()

# Simpan gambar + tampilkan
plt.savefig(os.path.join(splits_root, "confusion_matrix.png"))
plt.show()

# Save model
model_path = os.path.join(splits_root, "xgboost_latent.model")
model.save_model(model_path)
print(f"Model saved to {model_path}")
