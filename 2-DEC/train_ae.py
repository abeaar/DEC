import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from obspy import read

# ======================
# Config
# ======================
train_manifest = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits\train.csv"
val_manifest = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\splits\val.csv"

batch_size = 64
epochs = 30
lr = 1e-3
latent_dim = 128
input_length = 3001  # panjang waveform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

        # Padding / trimming agar pas 3001 sample
        if len(data) < input_length:
            pad_len = input_length - len(data)
            data = np.pad(data, (0, pad_len), mode="constant")
        else:
            data = data[:input_length]

        # Add channel dim (1, L)
        data = torch.tensor(data).unsqueeze(0)
        return data, data  # input = target (autoencoder)

# ======================
# Simple CAE Model
# ======================
class SimpleCAE(nn.Module):
    def __init__(self, latent_dim=128, input_length=3001):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),   # 3001 -> ~1500
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),   # ~1500 -> ~750
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),   # ~750 -> ~375
            nn.Dropout(0.2),
        )

        # Hitung ukuran output encoder secara otomatis
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            conv_out = self.encoder(dummy)
            self.conv_out_shape = conv_out.shape  # (1, C, L)
            conv_out_size = conv_out.numel()

        # Fully connected bottleneck
        self.fc_enc = nn.Linear(conv_out_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, conv_out_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        z = self.fc_enc(h_flat)

        # Decoder
        d_fc = self.fc_dec(z)
        d_unflat = d_fc.view(h.size(0), self.conv_out_shape[1], self.conv_out_shape[2])
        out = self.decoder(d_unflat)

        # Trim/Pad output supaya sama persis dengan input
        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
        elif out.size(2) < x.size(2):
            pad_len = x.size(2) - out.size(2)
            out = torch.nn.functional.pad(out, (0, pad_len))

        return out, z

# ======================
# Training Loop
# ======================
def train_autoencoder():
    # Dataset & Dataloader
    train_ds = SeismicDataset(train_manifest)
    val_ds = SeismicDataset(val_manifest)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model, loss, optimizer
    model = SimpleCAE(latent_dim=latent_dim, input_length=input_length).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save model
    save_path = os.path.join(os.path.dirname(train_manifest), "simple_cae.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model

if __name__ == "__main__":
    train_autoencoder()
