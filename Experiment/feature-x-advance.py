import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP 
from sklearn.metrics import silhouette_score
import os # Untuk list file di directory

# --- IMPORT OBSPY ---
from obspy import read

# --- 1. LOAD DATA FROM MSEED FILES ---
data_dir = r"E:\Skripsi\DEC\dataset\zscore_normalized_2022"
file_list = os.listdir(data_dir) # Urutkan untuk konsistensi
file_list = [f for f in file_list if f.endswith('.mseed')] # Pastikan hanya file mseed

target_length = 3001  # Panjang target yang diinginkan
waveforms = []        # Inisialisasi list untuk menyimpan waveforms

print(f"Found {len(file_list)} MSEED files. Loading and standardizing length...")

for filename in file_list:
    filepath = os.path.join(data_dir, filename)
    try:
        st = read(filepath)
        tr = st[0]
        data = tr.data

        # STANDARDISASI PANJANG: Zero-Padding
        current_length = len(data)
        if current_length < target_length:
            # Hitung berapa banyak zeros yang perlu ditambahkan
            pad_width = target_length - current_length
            # Tambahkan zeros di akhir sinyal
            padded_data = np.pad(data, (0, pad_width), mode='constant')
            waveforms.append(padded_data)
        elif current_length == target_length:
            waveforms.append(data)
        else:
            # Jika lebih panjang, potong saja (truncate)
            # Peringatan: Kehilangan data!
            waveforms.append(data[:target_length])
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Konversi ke array numpy
waveforms = np.array(waveforms)
print(f"Successfully loaded {len(waveforms)} waveforms")
print(f"Shape of waveforms array: {waveforms.shape}") # (4000, 3001)

# --- 2. PREPROCESSING: WAVELET DENOISING ---
# (Karena data sudah dinormalisasi, langkah standard scaling mungkin tidak perlu)
# Namun denoising tetap bisa dilakukan

# def wavelet_denoise(signal, wavelet='db4', level=1):
#     """
#     Membersihkan noise dari sinyal menggunakan wavelet thresholding.
#     """
#     coeff = pywt.wavedec(signal, wavelet, level=level)
#     # Hitung threshold (misal, menggunakan universal threshold)
#     sigma = np.median(np.abs(coeff[-level])) / 0.6745
#     uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
#     coeff[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeff[1:]]
#     denoised_signal = pywt.waverec(coeff, wavelet)
#     return denoised_signal[:len(signal)] # Pastikan panjang output sama

# print("Applying wavelet denoising...")
# denoised_waveforms = np.array([wavelet_denoise(wf) for wf in waveforms])

# --- 3. FEATURE EXTRACTION: CONTINUOUS WAVELET TRANSFORM (CWT) ---
# Tentukan parameter-parameter CWT
frequencies = np.logspace(-1, 1.5, num=50, base=2) # Misal: 2^[-1 hingga 1.5], 50 titik
widths = pywt.frequency2scale('cmor', frequencies) # Untuk wavelet Complex Morlet

# Inisialisasi list untuk menyimpan scalogram
all_scalograms = []

print("Performing Continuous Wavelet Transform...")
for i, wf in enumerate(waveforms):
    if i % 500 == 0:
        print(f"Processing waveform {i}/{len(waveforms)}")
    
    # Hitung CWT
    coeffs, freqs = pywt.cwt(wf, scales=widths, wavelet='cmor')
    # Ambil magnitude dari koefisien kompleks
    scalogram = np.abs(coeffs)
    
    # Simpan scalogram untuk waveform ini
    all_scalograms.append(scalogram)

# Konversi ke array numpy: bentuk (4000, num_freqs, num_times)
all_scalograms = np.array(all_scalograms)
print(f"Scalograms shape: {all_scalograms.shape}")

# --- 4. FEATURE EXTRACTION: GLCM TEXTURE FEATURES ---
# Tentukan parameter GLCM
distances = [1, 3]      # Jarak piksel (coba beberapa jarak)
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Arah
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Inisialisasi list untuk menyimpan semua fitur
all_texture_features = []

print("Extracting GLCM texture features...")
for i, scalogram in enumerate(all_scalograms):
    if i % 500 == 0:
        print(f"Processing scalogram {i}/{len(all_scalograms)}")
    
    # Normalisasi scalogram ke range 0-255 untuk GLCM
    scalogram_min = scalogram.min()
    scalogram_max = scalogram.max()
    if scalogram_max > scalogram_min: # Hindari division by zero
        scalogram_norm = (255 * (scalogram - scalogram_min) / 
                         (scalogram_max - scalogram_min)).astype(np.uint8)
    else:
        scalogram_norm = np.zeros_like(scalogram, dtype=np.uint8)
    
    # Hitung GLCM
    glcm = graycomatrix(scalogram_norm,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    
    # Ekstrak properti dari GLCM
    feature_vector = []
    for prop in properties:
        feature_values = graycoprops(glcm, prop).flatten()
        feature_vector.extend(feature_values)
    
    all_texture_features.append(feature_vector)

# Konversi ke array numpy: bentuk (4000, num_features)
X_texture = np.array(all_texture_features)
print(f"Texture features shape: {X_texture.shape}")

# --- 5. DIMENSIONALITY REDUCTION: UMAP ---
print("Applying UMAP dimensionality reduction...")
scaler = StandardScaler()
X_texture_scaled = scaler.fit_transform(X_texture)

# Jalankan UMAP - bisa tuning parameter ini
reducer = UMAP(n_components=2, 
                    random_state=42, 
                    n_neighbors=30, 
                    min_dist=0.1,
                    metric='euclidean')
X_umap = reducer.fit_transform(X_texture_scaled)

# --- 6. CLUSTERING: K-MEANS ON UMAP EMBEDDING ---
print("Performing K-Means clustering...")
# Tentukan jumlah cluster optimal dengan elbow method
# Atau bisa juga menggunakan silhouette analysis
optimal_k = 6 # Contoh saja, perlu dicari yang optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_umap)

# --- 7. EVALUATION & VISUALIZATION ---
# Hitung Silhouette Score
score = silhouette_score(X_umap, cluster_labels)
print(f"Silhouette Score: {score:.3f}")

# Visualisasi hasil clustering di ruang UMAP
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=cluster_labels, 
                     cmap='viridis', 
                     s=15, 
                     alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title(f'UMAP Projection of Seismic Events\nSilhouette Score: {score:.3f}')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig('umap_clustering_result.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 8. ANALISIS HASIL CLUSTER ---
print("Analyzing cluster results...")
# Untuk setiap cluster, plot beberapa waveform contoh dan rata-ratanya
for cluster_id in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_size = len(cluster_indices)
    
    print(f"Cluster {cluster_id}: {cluster_size} events ({cluster_size/len(waveforms)*100:.1f}%)")
    
    # Plot waveform rata-rata cluster
    plt.figure(figsize=(12, 6))
    
    # Plot beberapa contoh acak
    if cluster_size > 10:
        sample_indices = np.random.choice(cluster_indices, size=10, replace=False)
        for idx in sample_indices:
            plt.plot(waveforms[idx], alpha=0.2, color='gray', linewidth=0.5)
    
    # Plot waveform rata-rata cluster
    average_waveform = np.mean(waveforms[cluster_indices], axis=0)
    plt.plot(average_waveform, linewidth=2.5, color='red', 
             label=f'Cluster {cluster_id} Average (n={cluster_size})')
    
    plt.title(f'Waveforms for Cluster {cluster_id}')
    plt.xlabel('Time Samples')
    plt.ylabel('Amplitude (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster_id}_waveforms.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 9. SIMPAN HASIL ---
# Simpan hasil clustering untuk analisis lebih lanjut
np.savez('clustering_results.npz',
         waveforms=waveforms,
        #  denoised_waveforms=denoised_waveforms,
         scalograms=all_scalograms,
         texture_features=X_texture,
         umap_embedding=X_umap,
         cluster_labels=cluster_labels,
         silhouette_score=score)

print("Analysis complete! Results saved to clustering_results.npz")