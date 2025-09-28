import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Path ke manifest dari cleaned dataset
clean_manifest = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\cleaned\manifest.csv"
# Path ke manifest dari augmented dataset
aug_manifest = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\augmented\manifest.csv"

# Path output untuk hasil split
output_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits"
os.makedirs(output_root, exist_ok=True)

print("Loading manifests...")
df_clean = pd.read_csv(clean_manifest)
df_aug = pd.read_csv(aug_manifest)

print(f"Clean dataset: {len(df_clean)} samples")
print(f"Augmented dataset: {len(df_aug)} samples")

# === Split hanya dari clean dataset ===
train_clean, test_clean = train_test_split(
    df_clean, test_size=0.15, stratify=df_clean['label'], random_state=42
)

train_clean, val_clean = train_test_split(
    train_clean, test_size=0.15 / 0.85, stratify=train_clean['label'], random_state=42
)

print("\n=== Split summary (clean only) ===")
print(f"Train clean: {len(train_clean)}")
print(f"Val clean:   {len(val_clean)}")
print(f"Test clean:  {len(test_clean)}")

# === Tambahkan augmented dataset hanya ke train ===
train_final = pd.concat([train_clean, df_aug], ignore_index=True)
val_final = val_clean.copy()
test_final = test_clean.copy()

print("\n=== Final dataset summary (with augmentation in train only) ===")
print(f"Train total: {len(train_final)} (clean + augmented)")
print(f"Val total:   {len(val_final)} (clean only)")
print(f"Test total:  {len(test_final)} (clean only)")

# === Simpan hasil split ===
train_final.to_csv(os.path.join(output_root, "train.csv"), index=False)
val_final.to_csv(os.path.join(output_root, "val.csv"), index=False)
test_final.to_csv(os.path.join(output_root, "test.csv"), index=False)

print(f"\nSplits saved in: {output_root}")
