import os
from obspy import read
from obspy.signal import filter
import matplotlib.pyplot as plt

def clean_seismic_data(input_folder, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Process each file
    file_count = 0
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.mseed'):
            file_count += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"\nProcessing {file_count}: {filename}")
            
            # Read file
            st = read(input_path)
            tr = st[0]  # Get first (and only) trace
            
            print(f"Original samples: {tr.stats.npts}")
            print(f"Sampling rate: {tr.stats.sampling_rate} Hz")
            
            # Detrend (remove linear trend)
            tr.detrend('linear')
            print("Applied: Linear detrend")
            
            # Bandpass filter (1-45 Hz)
            nyquist = tr.stats.sampling_rate / 2
            lowcut = 1.0
            highcut = min(15.0, nyquist * 0.9)
            tr.filter('bandpass', freqmin=lowcut, freqmax=highcut)
            print(f"Applied: Bandpass filter {lowcut}-{highcut} Hz")
            
            # Save cleaned data
            st.write(output_path, format='MSEED')
            print(f"Saved: {filename}")
            
            # Plot before/after comparison
            # plt.figure(figsize=(12, 6))
            
            # # Plot original (reload for comparison)
            # st_original = read(input_path)
            # plt.subplot(2, 1, 1)
            # plt.plot(st_original[0].data, 'b-', linewidth=0.5)
            # plt.title(f'Original - {filename}')
            # plt.ylabel('Amplitude')
            # plt.grid(True)
            
            # # Plot cleaned
            # plt.subplot(2, 1, 2)
            # plt.plot(tr.data, 'r-', linewidth=0.5)
            # plt.title(f'Cleaned - {filename}')
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.grid(True)
            
            # plt.tight_layout()
            # plt.show()
    
    print(f"\n=== Completed ===")
    print(f"Total files processed: {file_count}")
    print(f"Cleaned files saved in: {output_folder}")

if __name__ == "__main__":
    input_folder = r"E:\Skripsi\DEC\dataset\filtered_mseed_2022"
    output_folder = r"E:\Skripsi\DEC\dataset\clean_mseed_2022"
    
    clean_seismic_data(input_folder, output_folder)
