import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_squared_error
from datetime import datetime

# HANYA SEKALI - Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

class SeismicDataset(Dataset):
    """PyTorch Dataset untuk seismic windows"""
    def __init__(self, windows_normalized):
        #  PERBAIKAN: Pastikan data sudah normalized dengan benar
        self.data = torch.FloatTensor(windows_normalized)
        # Debug: Print data statistics
        print(f" Data Statistics:")
        print(f"  Min: {self.data.min():.6f}")
        print(f"  Max: {self.data.max():.6f}")
        print(f"  Mean: {self.data.mean():.6f}")
        print(f"  Std: {self.data.std():.6f}")
        
        # Warn if data looks problematic
        if self.data.std() < 0.1 or self.data.std() > 5.0:
            print("️  WARNING: Data std looks problematic! Consider re-normalization.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, sample

class OptimizedSeismicAutoencoder(nn.Module):
    """ OPTIMIZED Autoencoder architecture untuk seismic data"""
    def __init__(self, input_size=6000, encoding_dim=64):
        super(OptimizedSeismicAutoencoder, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        
        #  PERBAIKAN: Arsitektur yang lebih efisien dengan BatchNorm dan Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),  # Batch normalization untuk stabilitas
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, encoding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, input_size)
        )
        
        #  PERBAIKAN: Proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights dengan Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def load_preprocessed_data(data_dir='E:\\Skripsi\\DEC\\1-preparation\\preprocessed_data'):
    """Load data hasil dari step 2 dengan additional checks"""
    print(f" Loading preprocessed data from {data_dir}/")
    
    windows = np.load(os.path.join(data_dir, 'windows_normalized.npy'))
    
    #  PERBAIKAN: Additional data validation
    print(f" Data loaded successfully!")
    print(f" Windows shape: {windows.shape}")
    print(f" Data range: [{windows.min():.3f}, {windows.max():.3f}]")
    print(f" Data mean: {windows.mean():.6f}, std: {windows.std():.6f}")
    
    # Load metadata
    with open(os.path.join(data_dir, 'window_info.pickle'), 'rb') as f:
        window_info = pickle.load(f)
    with open(os.path.join(data_dir, 'scaler.pickle'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(data_dir, 'summary.pickle'), 'rb') as f:
        summary = pickle.load(f)
    
    print(f" Window info: {len(window_info)} entries")
    
    return windows, window_info, scaler, summary

def create_data_loaders(windows, batch_size=32, train_split=0.8):  #  Reduced batch size
    """Create PyTorch data loaders"""
    print(f" Creating data loaders...")
    
    dataset = SeismicDataset(windows)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f" Train samples: {len(train_dataset)}")
    print(f" Validation samples: {len(val_dataset)}")
    print(f" Batch size: {batch_size}")
    
    return train_loader, val_loader

def train_autoencoder_optimized(model, train_loader, val_loader, num_epochs=100):
    """ OPTIMIZED Training function"""
    print(f" Starting OPTIMIZED autoencoder training...")
    print(f"⏱️  Epochs: {num_epochs}")
    print(f" Device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    #  PERBAIKAN: Multiple optimizers and better scheduling
    criterion = nn.MSELoss()
    
    # Try different learning rates based on epoch
    initial_lr = 0.01  # 10x higher than original
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    #  PERBAIKAN: More aggressive scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 25
    stagnation_threshold = 0.001  # If improvement < 0.001, consider stagnant
    
    print(f" Initial Learning Rate: {initial_lr}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        #  Training Phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output, encoded = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            
            #  PERBAIKAN: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        #  Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output, encoded = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        #  PERBAIKAN: Better progress reporting
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train: {avg_train_loss:.6f} | '
                  f'Val: {avg_val_loss:.6f} | '
                  f'LR: {current_lr:.2e} | '
                  f'Best: {best_val_loss:.6f}')
        
        #  PERBAIKAN: Enhanced early stopping logic
        improvement = best_val_loss - avg_val_loss
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr
            }, 'models/best_autoencoder.pth')
            
            if improvement > stagnation_threshold:
                print(f" New best model! Improvement: {improvement:.6f}")
        else:
            patience_counter += 1
        
        #  PERBAIKAN: Dynamic intervention if stuck
        if epoch > 20 and avg_val_loss > 0.6:
            print("️  Model seems stuck at high loss!")
            print(f"   Current Val Loss: {avg_val_loss:.6f}")
            print(f"   Expected by now: <0.4")
            
            if epoch == 25:  # Intervention point
                print(" APPLYING LEARNING RATE BOOST!")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 3.0  # 3x boost
                print(f"   New LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break
        
        #  PERBAIKAN: Success check
        if avg_val_loss < 0.15:
            print(f" EXCELLENT! Reached target loss: {avg_val_loss:.6f}")
            break
    
    print(f"\n Training completed!")
    print(f" Best validation loss: {best_val_loss:.6f}")
    
    #  PERBAIKAN: Performance assessment
    if best_val_loss < 0.2:
        print(" EXCELLENT: Model trained successfully!")
    elif best_val_loss < 0.4:
        print("️  GOOD: Model is decent, but could be better.")
    else:
        print(" POOR: Model needs more work. Consider:")
        print("   - Different architecture")
        print("   - Better data preprocessing")
        print("   - Longer training")
    
    return train_losses, val_losses, best_val_loss

def main():
    """ OPTIMIZED Main function"""
    print(" Starting Step 3: OPTIMIZED Autoencoder Training")
    print("=" * 70)
    
    # Load data
    windows, window_info, scaler, summary = load_preprocessed_data()
    
    # Create data loaders with smaller batch size
    train_loader, val_loader = create_data_loaders(windows, batch_size=32)
    
    # Initialize OPTIMIZED model
    input_size = windows.shape[1]
    model = OptimizedSeismicAutoencoder(input_size=input_size, encoding_dim=64).to(device)  # Smaller encoding
    
    print(f" Model initialized:")
    print(f"   Input size: {input_size}")
    print(f"   Encoding dim: 64")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with optimized function
    train_losses, val_losses, best_val_loss = train_autoencoder_optimized(
        model, train_loader, val_loader, num_epochs=100
    )
    
    print(f"\n Step 3 Complete!")
    print(f" Model saved: models/best_autoencoder.pth")
    print(f" Best validation loss: {best_val_loss:.6f}")
    
    #  PERBAIKAN: Quick visualization if matplotlib available
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Target Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Convergence Check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Training plots saved to: models/training_progress.png")
    except:
        print("️  Plotting skipped (matplotlib not available)")
    
    return model

if __name__ == "__main__":
    model = main()
