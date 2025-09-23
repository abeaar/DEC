import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from obspy import read
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SeismicDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        self.filenames = []
        
        print("Loading seismic data...")
        skipped_count = 0
        loaded_count = 0
        
        for filename in sorted(os.listdir(data_folder)):
            if filename.endswith('.mseed'):
                file_path = os.path.join(data_folder, filename)
                st = read(file_path)
                tr = st[0]
                
                # Filter: hanya ambil data dengan >= 3001 samples
                if len(tr.data) >= 3001:
                    # Truncate ke exactly 3001 samples untuk konsistensi
                    data = torch.FloatTensor(tr.data[:3001])
                    self.data.append(data)
                    self.filenames.append(filename)
                    loaded_count += 1
                else:
                    skipped_count += 1
        
        if self.data:  # Only stack if we have data
            self.data = torch.stack(self.data)
            
        print(f"Loaded: {loaded_count} files")
        print(f"Skipped: {skipped_count} files (< 3001 samples)")
        print(f"Final data shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, alpha=1.0):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        # Cluster centers (learnable parameters)
        encoding_dim = list(autoencoder.encoder.children())[-1].out_features
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, encoding_dim))
    
    def forward(self, x):
        # Get encoded features
        _, encoded = self.autoencoder(x)
        
        # Compute soft assignment probabilities
        q = self.soft_assignment(encoded)
        
        return q, encoded
    
    def soft_assignment(self, encoded):
        # Student t-distribution
        distances = torch.cdist(encoded, self.cluster_centers)
        q = 1.0 / (1.0 + distances**2 / self.alpha)
        q = q**(self.alpha + 1) / 2
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        # Compute target distribution P
        weight = q**2 / torch.sum(q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()
        return p

def train_autoencoder(model, dataloader, epochs=150, lr=0.0005):
    """Pre-train autoencoder"""
    print("=== Pre-training Autoencoder ===")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            decoded, _ = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    print("Autoencoder pre-training completed!")
    return model

def initialize_cluster_centers(model, dataloader, n_clusters):
    """Initialize cluster centers using K-means"""
    print("=== Initializing Cluster Centers ===")
    
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, encoded = model(batch)
            features.append(encoded.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    return torch.FloatTensor(kmeans.cluster_centers_).to(device)

def train_dec(dec_model, dataloader, epochs=150, lr=0.0005, update_interval=10):
    """Train DEC model"""
    print("=== Training DEC ===")
    
    optimizer = optim.SGD(dec_model.parameters(), lr=lr, momentum=0.9)
    
    dec_model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # Update target distribution
        if epoch % update_interval == 0:
            dec_model.eval()
            q_all = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(device)
                    q, _ = dec_model(batch)
                    q_all.append(q)
            q_all = torch.cat(q_all, dim=0)
            p_all = dec_model.target_distribution(q_all)
            dec_model.train()
        
        # Training loop
        batch_idx = 0
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.size(0)
            
            # Get target distribution for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            p_batch = p_all[start_idx:end_idx]
            
            optimizer.zero_grad()
            q, _ = dec_model(batch)
            
            # KL divergence loss
            loss = nn.KLDivLoss(reduction='batchmean')(torch.log(q), p_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_idx += 1
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, KL Loss: {avg_loss:.6f}")
    
    print("DEC training completed!")
    return dec_model

def evaluate_clustering(dec_model, dataloader):
    """Evaluate and visualize clustering results"""
    print("=== Evaluating Results ===")
    
    dec_model.eval()
    all_features = []
    all_assignments = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            q, encoded = dec_model(batch)
            
            assignments = torch.argmax(q, dim=1)
            
            all_features.append(encoded.cpu().numpy())
            all_assignments.append(assignments.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    assignments = np.concatenate(all_assignments, axis=0)
    
    # Cluster statistics
    unique_clusters, counts = np.unique(assignments, return_counts=True)
    print(f"\nClustering Results:")
    print(f"Number of clusters found: {len(unique_clusters)}")
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(assignments)) * 100
        print(f"Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=assignments, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('DEC Clustering Results (t-SNE Visualization)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    
    return assignments, features

def visualize_cluster_examples(dataset, assignments, n_examples=5):
    """
    Visualize n examples from each cluster
    """
    print("\n=== Visualizing Cluster Examples ===")
    
    unique_clusters = np.unique(assignments)
    n_clusters = len(unique_clusters)
    
    # Create subplot grid
    fig, axes = plt.subplots(n_clusters, n_examples, figsize=(20, 4*n_clusters))
    fig.suptitle('Example Windows from Each Cluster', fontsize=16)
    
    for i, cluster in enumerate(unique_clusters):
        # Get indices of windows in this cluster
        cluster_indices = np.where(assignments == cluster)[0]
        
        # Randomly select n_examples
        if len(cluster_indices) >= n_examples:
            example_indices = np.random.choice(cluster_indices, n_examples, replace=False)
        else:
            example_indices = cluster_indices
            
        # Plot each example
        for j, idx in enumerate(example_indices):
            window = dataset.data[idx].cpu().numpy()
            axes[i, j].plot(window, 'k-', linewidth=0.5)
            axes[i, j].set_title(f'Cluster {cluster}\n{dataset.filenames[idx]}')
            axes[i, j].grid(True)
            
            # Remove axis labels for cleaner look
            axes[i, j].set_xticks([])
            if j == 0:  # Only show y-axis for leftmost plots
                axes[i, j].set_ylabel('Amplitude')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    # Configuration
    data_folder = r"E:\Skripsi\DEC\dataset\zscore_normalized_2022"
    n_clusters = 5# Adjust based on your expected number of seismic event types
    encoding_dim = 64
    batch_size = 64
    
    # Load dataset
    dataset = SeismicDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get input dimension (30s @ 100Hz = 3000 samples)
    input_dim = dataset.data.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize autoencoder
    autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
    
    # Pre-train autoencoder
    autoencoder = train_autoencoder(autoencoder, dataloader, epochs=150)
    
    # Initialize DEC
    dec_model = DEC(autoencoder, n_clusters).to(device)
    
    # Initialize cluster centers
    cluster_centers = initialize_cluster_centers(autoencoder, dataloader, n_clusters)
    dec_model.cluster_centers.data = cluster_centers
    
    # Train DEC
    dec_model = train_dec(dec_model, dataloader, epochs=150)
    
    # Evaluate results
    assignments, features = evaluate_clustering(dec_model, dataloader)
    visualize_cluster_examples(dataset, assignments)

    # Save results
    results = {
        'assignments': assignments,
        'features': features,
        'filenames': dataset.filenames
    }
    
    with open('dec_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save model
    torch.save(dec_model.state_dict(), 'dec_model.pth')
    
    print("\n=== DEC Training Complete ===")
    print(f"Results saved to: dec_results.pkl")
    print(f"Model saved to: dec_model.pth")

if __name__ == "__main__":
    main()
