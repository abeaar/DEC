import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from train_ae import SimpleCAE  

# ======================
# Config
# ======================
splits_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits"
model_path = os.path.join(splits_root, r"E:\Skripsi\DEC\2-DEC\simple_cae.pth")

batch_size = 64
latent_dim = 128
input_length = 3001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Dataset Class
# ======================
class SeismicDataset(Dataset):
    def __init__(self, manifest_csv):
        self.df = pd.read_csv(manifest_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        st = read(row["path"])
        tr = st[0]
        data = tr.data.astype(np.float32)
        # Add channel dim (1, L)
        data = torch.tensor(data).unsqueeze(0)
        return data, row["label"], row["filename"]

# ======================
# Load Model (Encoder only)
# ======================
model = SimpleCAE(latent_dim=latent_dim, input_length=input_length).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def extract_features(manifest_csv, output_csv):
    ds = SeismicDataset(manifest_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    features = []
    with torch.no_grad():
        for x, labels, filenames in loader:
            x = x.to(device)
            # Forward pass
            h = model.encoder(x)              # conv
            h_flat = h.view(h.size(0), -1)    # flatten
            z = model.fc_enc(h_flat)          # latent vector

            z = z.cpu().numpy()
            for i in range(len(z)):
                row = {"nama_file": filenames[i], "label": labels[i]}
                # tambahkan fitur f1..f128
                for j, val in enumerate(z[i]):
                    row[f"f{j+1}"] = val
                features.append(row)

    # Save to CSV
    df = pd.DataFrame(features)
    df.to_csv(output_csv, index=False)
    print(f"Saved latent features: {output_csv}")

# ======================
# Run extraction
# ======================
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        # path = os.path.join(splits_root, f"{split}.csv")
        # df = pd.read_csv(path)
        # before = len(df)
        # # Buang semua baris dengan label LF
        # df = df[df["label"] != "LF"]
        # after = len(df)
        # df.to_csv(path, index=False)
        # print(f"{split}.csv: {before} -> {after} (LF dihapus)")
        manifest_csv = os.path.join(splits_root, f"{split}.csv")
        output_csv = os.path.join(splits_root, f"{split}_latent.csv")
        extract_features(manifest_csv, output_csv)
