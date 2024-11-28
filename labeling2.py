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
    r, g, b = rgb_values
    if abs(r - g) < 30 and abs(g - b) < 30 and r > 200:  # Dekat dengan putih
        return "Dry Cargo"
    elif b > r and b > g:  # Dominan biru
        return "Frozen Cargo"
    elif g > r and g > b:  # Dominan hijau
        return "Agriculture-Related Cargo"
    elif r > g and r > b:  # Dominan merah
        return "Hazardous Cargo"
    elif r > 200 and g > 200 and b < 100:  # Kuning kekuningan
        return "Bulk Cargo"
    elif r > 100 and g > 50 and b < 50:  # Cokelat kekuningan
        return "Forest Industry Related"
    else:
        return "Uncategorized"

# Folder yang berisi gambar untuk data latih
image_folder = 'D:/SEMESTER5/ARTIFISIAL/datas/datas/containernumbers_container-serials/valid'  # Gantilah dengan folder gambar data latih

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
            "date_captured": "2023-02-10T13:38:47+00:00"  # Gantilah dengan tanggal yang sesuai
        })
        
        annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,  # Gantilah dengan kategori yang sesuai
            "bbox": [100, 100, 50, 50],  # Gantilah dengan bounding box yang sesuai
            "area": 2500,  # Gantilah dengan area yang sesuai
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
json_file_path = 'D:/SEMESTER5/ARTIFISIAL/datas/datas/containernumbers_container-serials/valid/annotations_with_color_labels2.json'
with open(json_file_path, 'w') as f:
    json.dump(annotations_data, f, indent=4)

print(f"JSON file with annotations saved to {json_file_path}")
