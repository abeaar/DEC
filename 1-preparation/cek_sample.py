import os
import glob
from obspy import read
from collections import Counter

FOLDER_PATH = 'E:\Skripsi\DEC\dataset\cleaned_mseed_2022-2' 

trace_length_counts = Counter()
total_files_processed = 0
total_traces_processed = 0

print(f"--- Menganalisis File MiniSEED di: {FOLDER_PATH} ---")
print("-" * 60)

search_pattern = os.path.join(FOLDER_PATH, f'*.{'mseed'}')
file = glob.glob(search_pattern)


for stream in file:
    try:
            # Menggunakan obspy.read() untuk membaca file
            # Setiap file .mseed dapat berisi satu atau lebih trace
        st = read(stream, format='mseed') 
        total_files_processed += 1
            
            # Iterasi melalui setiap trace dalam Stream
        for trace in st:
            current_length = len(trace.data)
            trace_length_counts[current_length] += 1
            total_traces_processed += 1
            
    except Exception as e:
            # Tangani file yang corrupt atau tidak terbaca
        print(f"Peringatan: Gagal membaca file {os.path.basename(stream)}. Error: {e}")

# Menyajikan frekuensi setiap panjang data yang ditemukan (dikelompokkan)
print("Frekuensi Kemunculan Setiap Panjang Data:")

sorted_lengths = sorted(trace_length_counts.keys())
    
for length in sorted_lengths:
    count = trace_length_counts[length]
    print(f"  Panjang {length} sampel: {count} trace")