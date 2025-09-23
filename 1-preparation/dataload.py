import os
from obspy import read, Stream, Trace
import matplotlib.pyplot as plt

def read_filtered_seismic(folder_path, output_folder='filtered_mseed'):
    """
    Read and filter seismic files, save matching traces to separate folder
    """
    filtered_data = []
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    print(f"Reading files from: {folder_path}")
    print("Filtering for: Channel=HHZ, Station=MEPAS")
    
    # Loop through all files in folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.msd'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the file
                st = read(file_path)
                
                # Filter traces
                for tr in st:
                    # Check if trace matches our criteria
                    if (tr.stats.channel == 'HHZ' and 
                        tr.stats.station == 'MEPAS'):
                        
                        # Create metadata dictionary
                        metadata = {
                            'filename': filename,
                            'station': tr.stats.station,
                            'channel': tr.stats.channel,
                            'starttime': tr.stats.starttime,
                            'endtime': tr.stats.endtime,
                            'sampling_rate': tr.stats.sampling_rate,
                            'npts': tr.stats.npts
                        }
                        
                        # Store data and metadata
                        filtered_data.append((tr.data, metadata))
                        
                        # Create new stream with single trace
                        new_stream = Stream(traces=[tr])
                        
                        # Generate output filename
                        output_filename = f"MEPAS_HHZ_{tr.stats.starttime.strftime('%Y%m%d_%H%M%S')}.mseed"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        # Save to mseed file
                        new_stream.write(output_path, format='MSEED')
                        
                        print(f"\nFound matching trace in: {filename}")
                        print(f"Saved to: {output_filename}")
                        print(f"Start time: {tr.stats.starttime}")
                        print(f"Duration: {tr.stats.endtime - tr.stats.starttime}")
            
                        
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total matching traces found: {len(filtered_data)}")
    print(f"Files saved in: {output_folder}")
    
    return filtered_data

if __name__ == "__main__":
    # Path to your dataset
    folder_path = r"E:\Skripsi\DEC\dataset\kating"
    output_folder = r"E:\Skripsi\DEC\dataset\filtered_mseed_2022"
    
    # Read and filter data
    filtered_data = read_filtered_seismic(folder_path, output_folder)
    
    # Example: Access the first trace data and metadata
    if filtered_data:
        first_data, first_metadata = filtered_data[0]
        print("\nFirst trace metadata:")
        for key, value in first_metadata.items():
            print(f"{key}: {value}")