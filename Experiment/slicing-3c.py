import pandas as pd
from obspy import read, UTCDateTime
import os
import matplotlib.pyplot as plt

# Baca file

st_combined = read(r"E:\Skripsi\DEC\dataset\2025\VG.MEPAC.00.HH*.D.2025.*.mseed") 
st_combined.merge(method=1)
st_combined.filter('bandpass', freqmin=0.5, freqmax=15, zerophase=True)

df = pd.read_excel(r"E:\Skripsi\DEC\Seismic_bulletin.xlsx")

# Buat folder output untuk data dan grafik
os.makedirs("output_gempa_3c", exist_ok=True)
os.makedirs("output_gempa_3c/plots", exist_ok=True)

print("=== INFO DATA MSEED ===")

print(f"Data gabungan: {st_combined[0].stats.starttime} hingga {st_combined[0].stats.endtime}")
print(f"Waktu mulai: {st_combined[0].stats.starttime}")
print(f"Waktu selesai: {st_combined[0].stats.endtime}")
print(f"Durasi data: {st_combined[0].stats.endtime - st_combined[0].stats.starttime} detik")

print("\n=== INFO GEMPA PERTAMA ===")
waktu_utama = UTCDateTime(df.iloc[0]['eventdate']) - 7*3600
microsecond = df.iloc[0]['eventdate_microsecond']
waktu_gempa = waktu_utama + (microsecond / 1000000.0)

print(f"Waktu gempa pertama: {waktu_gempa}")
print(f"Apakah waktu gempa dalam range data? {st_combined[0].stats.starttime <= waktu_gempa <= st_combined[0].stats.endtime}")

for index, row in df.iterrows():
    # Parse waktu
    # Convert WIB to UTC by subtracting 7 hours
    waktu_utama = UTCDateTime(row['eventdate']) - 7*3600  # 7 hours in seconds
    microsecond = row['eventdate_microsecond']
    duration = row['duration']
    
    # Hitung waktu gempa (tambah microsecond)
    waktu_gempa = waktu_utama + (microsecond / 1000000.0)
    
    # Tentukan window (2 detik sebelum + durasi gempa)
    start_time = waktu_gempa - 2
    end_time = waktu_gempa + duration
    
    # Potong dan simpan data
    st_trimmed = st_combined.copy().trim(start_time, end_time)
    
    # Nama file menggunakan eventid
    mseed_filename = f"output_gempa_3c/{row['eventid']}.mseed"
    plot_filename = f"output_gempa_3c/plots/{row['eventid']}.png"
    
    # Simpan data MSEED
    st_trimmed.write(mseed_filename, format="MSEED")
    
    fig, axes = plt.subplots(len(st_trimmed), 1, figsize=(12, 6), sharex=True)
    
    for i, trace in enumerate(st_trimmed):
        component = trace.stats.channel[-1] # Ambil huruf terakhir (Z, N, atau E)
        
        # Plot data per komponen
        axes[i].plot(trace.times(), trace.data, 'b-', linewidth=0.5)
        axes[i].set_title(f"Component: {component}", fontsize=10, loc='right')
        axes[i].grid(True)
        
        # Tampilkan label Amplitudo hanya pada subplot paling atas
        if i == 0:
             axes[i].set_ylabel('Amplitude (Counts)')
        
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f"Event ID: {row ['eventid']} - Origin Time (UTC): {waktu_gempa}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    
    print(f"Disimpan: {mseed_filename}")
    print(f"Plot disimpan: {plot_filename}")

print("Selesai! Semua gempa telah diexport dengan grafik.")
