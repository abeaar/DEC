import pandas as pd
from obspy import read, Stream
import os
import numpy as np 
import glob 
import matplotlib.pyplot as plt
import random 
from collections import Counter # Tetap berguna untuk statistik di masa depan

INPUT_DIR = r"E:\Skripsi\DEC\dataset\cleaned_mseed_2022-2" # Folder data input yang sudah CLEAN dan NORMALIZED
OUTPUT_DIR_AUGMENTED = r"E:\Skripsi\DEC\dataset\augmented_mseed_2022" # Folder untuk menyimpan data hasil augmentasi

TARGET_LENGTH = 3001 
MAX_SHIFT_SAMPLES = int(TARGET_LENGTH * 0.25) # +/- 5% pergeseran acak
FILE_EXTENSION = "mseed"

OUTPUT_DIR_AUGMENTED_PLOT = os.path.join(OUTPUT_DIR_AUGMENTED, "plot")
os.makedirs(OUTPUT_DIR_AUGMENTED_PLOT, exist_ok=True)
os.makedirs(OUTPUT_DIR_AUGMENTED, exist_ok=True)

print(f"Memulai Proses Data Augmentation (Random Time Shifting)... Target Shift")


def apply_random_shift(trace):
    """Menerapkan pergeseran waktu acak (Random Time Shifting) pada trace."""
    
    shift_amount = random.randint(-MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES)
    
    # Gunakan .copy() untuk memastikan trace asli tidak termodifikasi
    data = trace.data.copy()
    shifted_data = np.zeros_like(data)
    
    # Zero Filling: Mengisi ruang kosong (sampel yang "dibungkus") dengan nol
    if shift_amount > 0:
        # Shift ke kanan: data awal menjadi zeros, data dipindah ke kanan
        shifted_data[shift_amount:] = data[:-shift_amount]
    elif shift_amount < 0:
        # Shift ke kiri: data akhir menjadi zeros, data dipindah ke kiri
        shift_amount = abs(shift_amount)
        shifted_data[:-shift_amount] = data[shift_amount:]
    else:
        # Tidak ada shift
        shifted_data = data
        
    trace_augmented.data = shifted_data
    trace_augmented = trace.copy()
    
    trace_augmented.data = shifted_data
    
    return trace_augmented, shift_amount

mseed_files = glob.glob(os.path.join(INPUT_DIR, f"*.{FILE_EXTENSION}"))
plot_count = 0

for filename in mseed_files:
    # Baca file MiniSEED (Stream)
    st_original = read(filename)
    st_augmented = Stream() 
    # Tentukan nama file

    base_filename = os.path.basename(filename)
    # Tentukan nama file
    base_filename = os.path.basename(filename)
    file_name_only = os.path.splitext(base_filename)[0]
    
    new_filename = os.path.join(OUTPUT_DIR_AUGMENTED, base_filename)
    plot_path = os.path.join(OUTPUT_DIR_AUGMENTED_PLOT, f"{file_name_only}.png")
    
    
    print(f"\nProcessing file: {base_filename}")
    
    # Flag untuk Plot: Hanya plot trace pertama dari file ini
    
    for i, trace in enumerate(st_original):
        # 1. Terapkan Augmentasi (Random Time Shifting)

        # 1. Terapkan Augmentasi (Random Time Shifting)
        trace_aug, shift_amount = apply_random_shift(trace)
        st_augmented.append(trace_aug)
        
        # 2. PENYIMPANAN PLOT: Visualisasi trace pertama saja (i == 0)
        plt.figure(figsize=(10, 6))
            
        
        plt.plot(trace_aug.data, 'r-', linewidth=0.8, label=f'Shifted ({shift_amount} samples)')
        plt.title(f'After Random Shift - {trace.id}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
            
        plot_count += 1

    # 3. Tulis Stream hasil augmentasi ke folder output
    st_augmented.write(new_filename, format="MSEED")
    
    print(f"   Applied: Random Time Shift. Saved to {OUTPUT_DIR_AUGMENTED}")

print("-" * 60)
print("PROSES SELESAI.")
print(f"Total {plot_count} plot augmentasi telah disimpan di {OUTPUT_DIR_AUGMENTED_PLOT}")
print(f"Total {len(mseed_files)} file MiniSEED telah di-augmentasi dan disimpan.")