import numpy as np
from obspy import read


# Fungsi untuk membaca file .msd
def read_msd_file(file_path):
    st = read(file_path)
    for i in range(6):
        st[i].plot()
    signals = [tr.data for tr in st]  # Mengambil data dari setiap trace

    return signals

print(read_msd_file(r'E:\Skripsi\DEC\output_gempa\0a11ee6209d048289fa7aa4e18e2c29b.mseed'))