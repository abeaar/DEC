import obspy
from obspy.signal.trigger import classic_sta_lta
import matplotlib.pyplot as plt

# --- 1. MEMBACA FILE MSEED ---
# Ganti 'your_file.mseed' dengan nama file mseed Anda
st = obspy.read('your_file.mseed')
tr = st[0]

# --- 2. FILTERING DAN PREPROCESSING ---
tr.detrend('demean')
tr.filter('bandpass', freqmin=1.0, freqmax=15.0)

# --- 3. KONFIGURASI DAN APLIKASI STA/LTA ---
# Ambil parameter penting dari tabel
sta_window = 2.0  # seconds
lta_window = 20.0 # seconds
on_threshold = 1.7993
off_threshold = 0.9872

# Hitung nilai-nilai dalam jumlah sampel
sta_samples = int(sta_window * tr.stats.sampling_rate)
lta_samples = int(lta_window * tr.stats.sampling_rate)

# Hitung rasio STA/LTA
cft = classic_sta_lta(tr.data, sta_samples, lta_samples)

# --- 4. DETEKSI DAN PENYIMPANAN HASIL ---
# Mendapatkan waktu mulai event dari rasio
trigger_on = on_threshold
trigger_off = off_threshold
triggers = obspy.signal.trigger.trigger_onset(cft, trigger_on, trigger_off)

if triggers.size > 0:
    # Ambil event pertama yang terdeteksi
    first_trigger = triggers[0]
    start_index = first_trigger[0]
    end_index = first_trigger[1]

    # Ambil data seismik yang terdeteksi (trimming)
    start_time = tr.stats.starttime + start_index / tr.stats.sampling_rate
    end_time = tr.stats.starttime + end_index / tr.stats.sampling_rate
    
    # Trim data sekitar event
    trimmed_stream = st.trim(start_time - 5, end_time + 5)

    # Simpan hasil trim ke file baru
    output_filename = 'detected_event.mseed'
    trimmed_stream.write(output_filename, format='MSEED')
    print(f"Event terdeteksi dan disimpan ke '{output_filename}'")
else:
    print("Tidak ada event yang terdeteksi dalam data.")

# --- VISUALISASI (OPSIONAL) ---
# Anda bisa uncomment baris di bawah ini untuk melihat plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(cft, label='STA/LTA Ratio')
# ax.axhline(trigger_on, color='red', linestyle='--', label='On Threshold')
# ax.axhline(trigger_off, color='green', linestyle='--', label='Off Threshold')
# ax.set_title('STA/LTA Triggering')
# ax.set_xlabel('Sample Number')
# ax.set_ylabel('Ratio')
# ax.legend()
# plt.show()