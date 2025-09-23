import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read

def normalize_seismic_data(input_folder, output_folder, show_comparison=True):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    file_count = 0
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.mseed'):
            file_count += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing {file_count}: {filename}")
            
            # Read data
            st = read(input_path)
            tr = st[0]
            original_data = tr.data.copy()
            
            # Z-score normalization
            data = tr.data
            mean = np.mean(data)
            std = np.std(data)
            
            if std > 0:
                normalized_data = (data - mean) / std
            else:
                normalized_data = data - mean
            
            # Update trace data
            tr.data = normalized_data
            
            print(f"Original: mean={mean:.3f}, std={std:.3f}")
            print(f"Normalized: mean={np.mean(normalized_data):.3f}, std={np.std(normalized_data):.3f}")
            
            # Show comparison plot
            if show_comparison:
                plot_comparison(original_data, normalized_data, filename)
            
            # Save normalized data
            st.write(output_path, format='MSEED')
            print(f"Saved: {filename}\n")
    
    print(f"=== Completed ===")
    print(f"Total files normalized: {file_count}")

def plot_comparison(original, normalized, filename):
    """Plot before and after normalization comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Z-Score Normalization: {filename}', fontsize=14)
    
    # Flatten axes for easy access
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Time series plots
    time = np.arange(len(original))
    
    ax1.plot(time, original, 'b-', alpha=0.7)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time, normalized, 'r-', alpha=0.7)
    ax2.set_title('Normalized Data (Z-Score)')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Histograms
    ax3.hist(original, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_title('Original Distribution')
    ax3.set_xlabel('Amplitude')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(original), color='red', linestyle='--', 
                label=f'Mean: {np.mean(original):.3f}')
    ax3.legend()
    
    ax4.hist(normalized, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax4.set_title('Normalized Distribution')
    ax4.set_xlabel('Normalized Amplitude')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.mean(normalized), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(normalized):.3f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_folder = r"E:\Skripsi\DEC\dataset\clean_mseed_2022"
    output_folder = r"E:\Skripsi\DEC\dataset\zscore_normalized_2022"
    
    # Set show_comparison=False untuk skip plotting
    normalize_seismic_data(input_folder, output_folder, show_comparison=False)
