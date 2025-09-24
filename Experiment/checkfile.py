import numpy as np
import matplotlib.pyplot as plt

# Load file .npy
data = np.load("output_gempa_dec_ready/namafile.npy")
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

channels = ['Z', 'N', 'E']
colors = ['red', 'blue', 'green']

for i in range(3):
    axes[i].plot(data[i], color=colors[i], linewidth=0.8)
    axes[i].set_title(f'Channel {channels[i]}')
    axes[i].set_ylabel('Amplitude')
    axes[i].grid(True, alpha=0.3)
    
    # Tampilkan stats
    axes[i].text(0.02, 0.95, f'μ={np.mean(data[i]):.3f}, σ={np.std(data[i]):.3f}', 
                transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor="wheat"))

axes[2].set_xlabel('Time Points')
plt.suptitle('DEC Ready Data - 3 Components')
plt.tight_layout()
plt.show()