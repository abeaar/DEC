import pandas as pd
from obspy import read, UTCDateTime
import os
import matplotlib.pyplot as plt

# Baca file
st1 = read(r"E:\Skripsi\DEC\dataset\VG.MEPAC.00.HHZ.D.2025.254.mseed")
st2 = read(r"E:\Skripsi\DEC\dataset\VG.MEPAC.00.HHZ.D.2025.255.mseed")
st3 = read(r"E:\Skripsi\DEC\dataset\VG.MEPAC.00.HHZ.D.2025.256.mseed")
st4 = read(r"E:\Skripsi\DEC\dataset\VG.MEPAC.00.HHZ.D.2025.257.mseed")

st_combined = st1 + st2 + st3 + st4
st_combined.merge(method=1)
st_combined.filter('bandpass', freqmin=0.5, freqmax=15, zerophase=True)

df = pd.read_excel(r"E:\Skripsi\DEC\Seismic_bulletin.xlsx")

# Buat folder output untuk data dan grafik
os.makedirs("output_gempa", exist_ok=True)
os.makedirs("output_gempa/plots", exist_ok=True)

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
    mseed_filename = f"output_gempa/{row['eventid']}.mseed"
    plot_filename = f"output_gempa/plots/{row['eventid']}.png"
    
    # Simpan data MSEED
    st_trimmed.write(mseed_filename, format="MSEED")
    
    # Buat dan simpan plot
    plt.figure(figsize=(12, 4))
    plt.plot(st_trimmed[0].times(), st_trimmed[0].data, 'b-', linewidth=0.5)
    plt.title(f"Event ID: {row['eventid']} - {waktu_gempa}")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Disimpan: {mseed_filename}")
    print(f"Plot disimpan: {plot_filename}")

print("Selesai! Semua gempa telah diexport dengan grafik.")
