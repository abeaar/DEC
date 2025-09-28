import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Trace, Stream

# Path input (hasil cleaning)
input_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\cleaned"
# Path output untuk hasil augmentasi
output_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\augmented"
# File manifest metadata
manifest_path = os.path.join(output_root, "manifest.csv")

os.makedirs(output_root, exist_ok=True)

records = []
saved_count = 0

print(f"Reading from: {input_root}")

for label in sorted(os.listdir(input_root)):
    label_folder = os.path.join(input_root, label)
    if not os.path.isdir(label_folder):
        continue
    
    out_label_folder = os.path.join(output_root, label)
    os.makedirs(out_label_folder, exist_ok=True)
    
    for filename in sorted(os.listdir(label_folder)):
        if filename.endswith(".mseed"):
            file_path = os.path.join(label_folder, filename)
            st = read(file_path)
            
            for tr in st:
                data = tr.data.astype(np.float32)
                N = len(data)
                
                # Hitung batas shift (25% panjang sinyal)
                max_shift = int(0.25 * N)
                shift = np.random.randint(-max_shift, max_shift + 1)
                
                # Geser sinyal (dengan np.roll = 0)
                aug_data = np.zeros_like(data)
                if shift > 0:
                    aug_data[shift:] = data[:-shift]   # geser ke kanan
                elif shift < 0:
                    aug_data[:shift] = data[-shift:]   # geser ke kiri
                else:
                    aug_data = data.copy()

                # Buat trace baru
                aug_tr = Trace(data=aug_data, header=tr.stats)
                aug_st = Stream(traces=[aug_tr])
                
                # Simpan hasil augmentasi
                base_name = os.path.splitext(filename)[0]
                aug_name = f"{base_name}_shift.mseed"
                output_path = os.path.join(out_label_folder, aug_name)
                aug_st.write(output_path, format="MSEED")
                
                # Simpan plot waveform
                plot_name = f"{base_name}_shift.png"
                plot_path = os.path.join(out_label_folder, plot_name)
                plt.figure(figsize=(8, 3))
                plt.plot(np.arange(len(aug_data)), aug_data, color="blue", linewidth=0.8)
                plt.title(f"{label} | {aug_name} | shift={shift}")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                
                # Simpan metadata
                records.append({
                    "filename": aug_name,
                    "label": label,
                    "path": output_path,
                    "plot": plot_path,
                    "station": tr.stats.station,
                    "channel": tr.stats.channel,
                    "starttime": tr.stats.starttime,
                    "endtime": tr.stats.endtime,
                    "sampling_rate": tr.stats.sampling_rate,
                    "npts": len(aug_data),
                    "shift_samples": shift
                })
                
                saved_count += 1
                print(f"Saved augmented: {output_path}")

# Simpan manifest CSV
df = pd.DataFrame(records)
df.to_csv(manifest_path, index=False)

print("\n=== Summary ===")
print(f"Total augmented traces saved: {saved_count}")
print(f"Manifest saved at: {manifest_path}")
