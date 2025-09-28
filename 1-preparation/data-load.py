import os
from obspy import read, Stream

folder_path = r"E:\Skripsi\DEC\dataset\kating"
output_root = r"E:\Skripsi\DEC\dataset\ae-supervised-dataset"
os.makedirs(output_root, exist_ok=True)

print(f"Reading files from: {folder_path}")
    
saved_count = 0
    
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.msd'):
        file_path = os.path.join(folder_path, filename)
        st = read(file_path)
            
        for tr in st:
            if tr.stats.channel == 'HHZ' and tr.stats.station == 'MEPAS':
                # Ambil label dari nama file 
                label = filename.split('-')[-1].replace('.msd', '')

                if label in ['VTA', 'VTB']:
                    label = 'VT'  

                # Buat folder label jika belum ada
                label_folder = os.path.join(output_root, label)
                os.makedirs(label_folder, exist_ok=True)
                    
                # Simpan file dengan nama asli
                output_path = os.path.join(label_folder, filename)
                Stream(traces=[tr]).write(output_path, format='msd')
                    
                saved_count += 1
                print(f"Saved: {output_path}")
    
print("\n=== Summary ===")
print(f"Total traces saved: {saved_count}")
print(f"Files organized in: {output_root}")


