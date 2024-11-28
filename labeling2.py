import os
import json
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.cluster import KMeans
from PIL import Image

# Fungsi untuk menghitung warna dominan menggunakan K-Means
def get_dominant_color(image_path, k=3):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((50, 50))  # Resize for consistent processing
    pixels = np.array(resized_image).reshape(-1, 3)  # Flatten image to pixels
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]  # Ambil klaster paling dominan
    return dominant_color

# Fungsi untuk memetakan RGB ke label berdasarkan rentang warna yang telah ditentukan
def map_color_to_label(rgb_values):
    if rgb_values is None:
        return "Uncategorized"
    
    r, g, b = rgb_values
    if max(r, g, b) < 50:  # Hitam (nilai RGB rendah)
        return "Black"
    elif min(r, g, b) > 200:  # Putih (nilai RGB tinggi)
        return "White"
    elif r > g and r > b:  # Dominan merah
        return "Red"
    elif g > r and g > b:  # Dominan hijau
        return "Green"
    elif b > r and b > g:  # Dominan biru
        return "Blue"
    elif r > 200 and g > 200 and b < 100:  # Kuning
        return "Yellow"
    elif r > 100 and g > 50 and b < 50:  # Cokelat
        return "Brown"
    else:  # Jika tidak cocok, klasifikasikan sebagai "Other"
        return "Other"

# Folder yang berisi gambar untuk data latih
image_folder = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/data/containernumbers_container-serials/valid'  # Gantilah dengan folder gambar data latih

# Membaca gambar dan memberikan label warna dominan
image_files = os.listdir(image_folder)
annotations = []
images_info = []

for idx, image_filename in enumerate(image_files):
    image_path = os.path.join(image_folder, image_filename)
    
    if os.path.exists(image_path):
        # Hitung warna dominan
        dominant_color = get_dominant_color(image_path)
        label = map_color_to_label(dominant_color)
        
        # Menyimpan informasi gambar dan anotasi
        images_info.append({
            "id": idx,
            "file_name": image_filename,
            "height": 416,  # Gantilah sesuai dengan ukuran gambar jika diperlukan
            "width": 416,   # Gantilah sesuai dengan ukuran gambar jika diperlukan
        })
        
        annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,  # Gantilah dengan kategori yang sesuai
            "color_label": label,
            "dominant_color": dominant_color.tolist(),
            "segmentation": [],
            "iscrowd": 0
        })

# Membuat dictionary untuk menyimpan semua data
annotations_data = {
    'images': images_info,
    'annotations': annotations,
    'categories': [
        {"id": 1, "name": "Cargo", "supercategory": "none"}
    ]
}

# Menyimpan ke dalam file JSON
json_file_path = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/data/containernumbers_container-serials/valid/annotations_color_valid.json'
with open(json_file_path, 'w') as f:
    json.dump(annotations_data, f, indent=4)

print(f"JSON file with annotations saved to {json_file_path}")
