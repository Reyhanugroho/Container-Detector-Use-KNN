import os
import json
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from PIL import Image
import colorsys

# Fungsi untuk menghitung warna dominan menggunakan K-Means
def get_dominant_color(image_path, k=5):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((50, 50))  # Resize untuk pengolahan yang konsisten
    pixels = np.array(resized_image).reshape(-1, 3)  # Ubah gambar menjadi data piksel
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Menentukan warna dominan berdasarkan cluster dengan jumlah piksel terbesar
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    
    return dominant_color

# Fungsi untuk memetakan RGB ke label berdasarkan rentang warna yang telah ditentukan
def map_color_to_label(rgb_values):
    if rgb_values is None:
        return "Uncategorized"
    
    r, g, b = rgb_values
    r, g, b = r / 255.0, g / 255.0, b / 255.0  # Normalisasi warna
    
    # Konversi RGB ke HSV untuk mendeteksi warna secara lebih akurat
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    if v < 0.2:  # Gelap, kemungkinan hitam
        return "Black"
    elif s < 0.2:  # Tidak saturasi, kemungkinan putih atau abu-abu
        return "White"
    elif h >= 0.0 and h <= 0.1:  # Ranges for Red
        return "Red"
    elif h >= 0.1 and h <= 0.4:  # Ranges for Green
        return "Green"
    elif h >= 0.5 and h <= 0.75:  # Ranges for Blue
        return "Blue"
    elif h >= 0.1 and h <= 0.2 and v > 0.6:  # Ranges for Yellow
        return "Yellow"
    elif h >= 0.05 and h <= 0.13:  # Ranges for Brown
        return "Brown"
    else:
        return "Other"

# Folder yang berisi gambar untuk data latih
image_folder = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/Container-Detector-Use-KNN/data'  # Gantilah dengan folder gambar data latih

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
json_file_path = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/Container-Detector-Use-KNN/data/annotations_color.json'
with open(json_file_path, 'w') as f:
    json.dump(annotations_data, f, indent=4)

print(f"JSON file with annotations saved to {json_file_path}")