import obspy
import matplotlib.pyplot as plt
import numpy as np

# Ganti 'nama_file_anda.msd' dengan nama file yang ingin Anda baca.
file_msd = r'E:\Skripsi\DEC\dataset\VG.MEPAC.00.HHZ.D.2025.254.mseed'

try:
    # Fungsi read() dari ObsPy secara otomatis mengenali format MiniSEED/MSD
    # dan mengembalikan objek Stream
    stream = obspy.read(file_msd)

    # Menampilkan ringkasan data yang telah dibaca
    print("Ringkasan data:")
    print(stream)

    # Mengakses data gelombang seismik dari trace (jejak) pertama
    # dan menampilkan informasi statistik
    if len(stream) > 0:
        trace = stream[0]
        
        # --- Bagian Baru: Normalisasi Z-Score ---
        # Mengakses data mentah dari trace
        data = trace.data.astype(np.float32) # Mengubah tipe data untuk presisi yang lebih baik

        # Menghitung nilai rata-rata (mean) dan deviasi standar (std) dari data
        data_mean = data.mean()
        data_std = data.std()

        # Rumus normalisasi Z-Score
        zscore_normalized_data = (data - data_mean) / data_std
        
        # --- Akhir Bagian Normalisasi ---

        # Plot data asli dan data yang sudah dinormalisasi secara berdampingan
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot data asli
        ax1.plot(data)
        ax1.set_title(f"Waveform Asli - {trace.stats.station}")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        
        # Plot data yang sudah dinormalisasi Z-Score
        ax2.plot(zscore_normalized_data)
        ax2.set_title(f"Waveform Setelah Normalisasi Z-Score - {trace.stats.station}")
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Normalized Amplitude (Z-Score)")
        ax2.grid(True)
        
        print("\nInformasi trace pertama:")
        print(f"Stasiun: {trace.stats.station}")
        print(f"Durasi: {trace.stats.endtime - trace.stats.starttime} detik")
        print(f"Laju sampel: {trace.stats.sampling_rate} Hz")
        print(f"Jumlah sampel: {trace.stats.npts}")
        print(f"Nilai Rata-Rata Asli: {data_mean:.2f}, Deviasi Standar Asli: {data_std:.2f}")
        print(f"Nilai Rata-Rata Normalisasi: {zscore_normalized_data.mean():.2f}, Deviasi Standar Normalisasi: {zscore_normalized_data.std():.2f}")

        plt.tight_layout() # Mengatur tata letak agar tidak tumpang tindih
        plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_msd}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi kesalahan saat membaca file: {e}")