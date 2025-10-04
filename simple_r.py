import numpy as np
from obspy import read


# Fungsi untuk membaca file .msd
def read_msd_file(file_path):
    st = read(file_path)
    for i in range(6):
        st[i].plot()
    signals = [tr.data for tr in st]  # Mengambil data dari setiap trace

    return signals


def test_shift():
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Data asli:    {test_data}")
    
    # Shift +2
    shifted = np.zeros_like(test_data)
    shifted[2:] = test_data[:8]  # 10 - 2 = 8
    print(f"Shift +2:     {shifted}")
    
    # Shift -3  
    shifted2 = np.zeros_like(test_data)
    shifted2[:7] = test_data[3:]  # 10 - 3 = 7
    print(f"Shift -3:     {shifted2}")

# print(read_msd_file(r'E:\Skripsi\DEC\dataset\ae-supervised-dataset\augmented\GASBURST\2022-08-01_13-38-55-GASBURST_shift.mseed'))
print(read_msd_file(r'E:\Skripsi\DEC\Experiment\output_gempa\MP\0f9811266ed4418babcacecf4aac622a_MP.mseed'))
# test_shift()