import obspy
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# --- 1. Muat Data & Preprocessing ---
data_folder = 'E:\Skripsi\DEC\dataset\zscore_normalized_2022'
mseed_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.mseed')]

print(f"Memuat {len(mseed_files)} file dari folder '{data_folder}'...")

all_traces_data = []
# Set the expected length
expected_length = 3001

for file in tqdm(mseed_files, desc="Processing files"):
    st = obspy.read(file)
    for tr in st:
        # Check if the trace has the correct number of samples
        if tr.stats.npts == expected_length:
            all_traces_data.append(tr.data)
        else:
            print(f"Warning: Skipping file {file} with unexpected length of {tr.stats.npts} samples.")

if not all_traces_data:
    print("No valid data found with the correct number of samples. Please check your data files.")
    exit()

# Stack all valid traces into a single NumPy array
dataset_np = np.stack(all_traces_data)


# --- 2. Ekstraksi Fitur dengan PCA ---
n_components = 10 
print(f"\nMelakukan PCA untuk mereduksi dimensi data ke {n_components} komponen...")
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(dataset_np)

# --- 3. Klastering DBSCAN ---
# eps: Jarak maksimum antar dua sampel agar dianggap berada dalam satu lingkungan
# min_samples: Jumlah sampel minimum dalam sebuah lingkungan agar dianggap klaster
# Anda mungkin perlu menyesuaikan nilai-nilai ini
eps_value = 0.5 
min_samples_value = 5 

print(f"\nMelakukan klastering DBSCAN dengan eps={eps_value} dan min_samples={min_samples_value}...")
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = dbscan.fit_predict(pca_features)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_noise = list(labels).count(-1)
print(f"Klastering DBSCAN selesai. Ditemukan {num_clusters} klaster dan {num_noise} titik outlier (noise).")

# --- 4. Visualisasi Hasil Klastering ---
print("\nMemvisualisasikan hasil...")
pca_2d = PCA(n_components=2)
pca_features_2d = pca_2d.fit_transform(pca_features)

plt.figure(figsize=(10, 8))
# Titik dengan label -1 adalah outlier (noise)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Warnai outlier sebagai hitam
        col = 'k'
        marker = '.'
        size = 10
    else:
        marker = 'o'
        size = 20
    
    class_member_mask = (labels == k)
    xy = pca_features_2d[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=col, markeredgecolor='k', markersize=size)

plt.title(f'Hasil DBSCAN Clustering pada Fitur PCA\n({num_clusters} Klaster, {num_noise} Outlier)')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.show()

print("Visualisasi selesai. Plot telah ditampilkan.")