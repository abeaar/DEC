import os
import pandas as pd
import numpy as np
from obspy import read
import matplotlib.pyplot as plt

# Path input (hasil folderisasi per label)
input_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset"
# Path output untuk hasil cleaning
output_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset\cleaned"
# File manifest metadata
manifest_path = os.path.join(output_root, "manifest.csv")

os.makedirs(output_root, exist_ok=True)

records = []
saved_count = 0

print(f"Reading from: {input_root}")

for label in os.listdir(input_root):
    label_folder = os.path.join(input_root, label)
    
    # Folder output untuk label
    out_label_folder = os.path.join(output_root, label)
    os.makedirs(out_label_folder, exist_ok=True)
    
    for filename in os.listdir(label_folder):
        if filename.endswith(".msd"):
            file_path = os.path.join(label_folder, filename)
            st = read(file_path)
            
            for tr in st:
                # 1) Detrend + demean
                tr.detrend("linear")
                tr.detrend("demean")
                
                # 2) Bandpass filter
                tr.filter("bandpass", freqmin=0.5, freqmax=15.0, corners=4, zerophase=True)
                
                # 3) Normalisasi (z-score)
                data = tr.data.astype(np.float32)
                if np.std(data) > 0:
                    data = (data - np.mean(data)) / np.std(data)
                else:
                    data = np.zeros_like(data)
                # 4) Padding minimal 3001 sample
                if len(data) < 3001:
                    pad_len = 3001 - len(data)
                    data = np.pad(data, (0, pad_len), mode="constant")
                tr.data = data

                # Simpan hasil cleaning
                base_name = os.path.splitext(filename)[0] + ".mseed"
                output_path = os.path.join(out_label_folder, base_name)
                tr.write(output_path, format="MSEED")

                # Simpan plot waveform
                plot_filename = filename.replace(".msd", ".png")
                plot_path = os.path.join(out_label_folder, plot_filename)
                plt.figure(figsize=(8, 3))
                plt.plot(np.arange(len(tr.data)), tr.data, color="black", linewidth=0.8)
                plt.title(f"{label} | {filename}")
                plt.xlabel("Samples")
                plt.ylabel("Normalized Amplitude")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()

                # Simpan metadata untuk manifest
                records.append({
                    "filename": filename,
                    "label": label,
                    "path": output_path,
                    "station": tr.stats.station,
                    "channel": tr.stats.channel,
                    "starttime": tr.stats.starttime,
                    "endtime": tr.stats.endtime,
                    "sampling_rate": tr.stats.sampling_rate,
                    "npts": tr.stats.npts
                })
                
                saved_count += 1
                print(f"Saved cleaned: {output_path}")

# Buat manifest CSV
df = pd.DataFrame(records)
df.to_csv(manifest_path, index=False)

print("\n=== Summary ===")
print(f"Total cleaned traces saved: {saved_count}")
print(f"Manifest saved at: {manifest_path}")
