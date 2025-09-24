import pandas as pd
from obspy import read
import os
import numpy as np
import glob 
import matplotlib.pyplot as plt

# --- Konfigurasi Folder --- 
# Buat folder output
OUTPUT_DIR_NORM = "output_gempa_3c_norm"
OUTPUT_DIR_NORM_PLOT = os.path.join(OUTPUT_DIR_NORM, "plot")

os.makedirs(OUTPUT_DIR_NORM_PLOT, exist_ok=True)
os.makedirs(OUTPUT_DIR_NORM, exist_ok=True)


print("Memulai Normalisasi Z-Score...")

# Cari semua file MiniSEED di folder input
mseed_files = glob.glob(os.path.join("output_gempa_3c", "*.mseed"))

for filename in mseed_files:
    # 1. Baca file MiniSEED 3-komponen
    st = read(filename)
    
    # 2. Iterasi melalui setiap Trace (Z, N, E)
    for trace in st:
        
        # Ambil data numpy dari trace
        data = trace.data
        
        # Hitung rata-rata (mean) dan deviasi standar (std)
        mean = data.mean()
        std = data.std()
        
        # Lakukan Normalisasi Z-Score
        # Jika deviasi standar adalah 0 (trace datar), hindari pembagian nol
        if std != 0:
            trace.data = (data - mean) / std
        else:
            # Jika trace datar, biarkan nilainya nol
            trace.data = np.zeros_like(data)
    
    # 3. Tentukan nama file output yang baru
    base_filename = os.path.basename(filename)
    file_name_only = os.path.splitext(base_filename)[0]
    
    new_filename = os.path.join(OUTPUT_DIR_NORM, base_filename)
    plot_filename = os.path.join(OUTPUT_DIR_NORM_PLOT, f"{file_name_only }.png")
    
    # 4. Simpan Stream (3 Trace) yang sudah dinormalisasi
    st.write(new_filename, format="MSEED")
    
    fig, axes = plt.subplots(len(st), 1, figsize=(10, 6), sharex=True)
    
    for i, trace in enumerate(st):
        component = trace.stats.channel[-1] # Ambil huruf terakhir (Z, N, atau E)
        
        # Plot data yang sudah dinormalisasi
        axes[i].plot(trace.times(), trace.data, 'b-', linewidth=0.5)
        
        # Atur label
        axes[i].set_title(f"Component: {component} (Z-Score)", fontsize=10, loc='right')
        axes[i].grid(True)
        axes[i].set_ylim(-6, 6) # Atur batas Y agar terlihat jelas (most data is within -5 to 5)
        
        if i == 0:
             axes[i].set_ylabel('Amplitude ($\sigma$ Units)')
             
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f"Event ID: {file_name_only} - Z-Score Normalized", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_filename, dpi=200)
    plt.close(fig)

    print(f" Dinormalisasi dan disimpan: {new_filename}")

print("\nSelesai! Semua file telah dinormalisasi Z-Score.")