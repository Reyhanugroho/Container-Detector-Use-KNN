import os
import json
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import colorsys


# Fungsi untuk menghitung warna dominan berdasarkan radius titik tengah gambar
def get_dominant_color(image_path, k=5, radius=50):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # Mendefinisikan area di sekitar titik tengah gambar
    left = max(center_x - radius, 0)
    upper = max(center_y - radius, 0)
    right = min(center_x + radius, width)
    lower = min(center_y + radius, height)

    # Memotong gambar untuk mendapatkan area sekitar tengah
    cropped_image = image.crop((left, upper, right, lower))
    pixels = np.array(cropped_image).reshape(-1, 3)

    # Menggunakan K-Means untuk menemukan warna dominan
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Menentukan warna dominan
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]

    return dominant_color


# Fungsi untuk memetakan RGB ke label berdasarkan rentang warna yang telah ditentukan
def map_color_to_label(rgb_values):
    if rgb_values is None:
        return "Uncategorized"

    # Normalisasi nilai RGB ke rentang [0, 1]
    r, g, b = rgb_values
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # Konversi RGB ke HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Rentang warna yang diperluas
    if v > 0.9 and s < 0.2:  # Putih (nilai tinggi, saturasi rendah)
        return "White"
    elif h >= 0.0 and h <= 0.05 and s > 0.4:  # Merah (rentang hue diperluas untuk merah gelap)
        return "Red"
    elif h >= 0.05 and h <= 0.4 and s > 0.4:  # Hijau (rentang hijau diperluas untuk mencakup hijau kekuningan)
        return "Green"
    elif h >= 0.1 and h <= 0.2 and v > 0.6 and s > 0.5:  # Kuning (rentang hue diperluas ke sisi hijau dan merah)
        return "Yellow"
    elif h >= 0.5 and h <= 0.75 and s > 0.4:  # Biru (rentang hue tetap)
        return "Blue"
    else:
        return "Other"

# Folder yang berisi gambar untuk data latih
image_folder = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/Container-Detector-Use-KNN/data'

# Validasi folder
if not os.path.exists(image_folder):
    print(f"Folder {image_folder} tidak ditemukan.")
    exit()

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
combined_data = []

for idx, image_filename in enumerate(image_files):
    image_path = os.path.join(image_folder, image_filename)

    # Hitung warna dominan berdasarkan area tengah gambar
    dominant_color = get_dominant_color(image_path, radius=50)
    label = map_color_to_label(dominant_color)

    # Gabungkan nama gambar dengan anotasi dalam satu entri
    combined_data.append({
        "file_name": image_filename,
        "height": 416,  # Gantilah sesuai dengan ukuran gambar jika diperlukan
        "width": 416,   # Gantilah sesuai dengan ukuran gambar jika diperlukan
        "annotations": {
            "id": idx,
            "category_id": 1,  # Gantilah dengan kategori yang sesuai
            "color_label": label,
            "dominant_color": dominant_color.tolist() if dominant_color is not None else [],
            "segmentation": [],
            "iscrowd": 0
        }
    })

# Menyimpan ke dalam file JSON
output_path = 'D:/Serba Serbi Kuliah/MATERI KULIAH SMT 5/AI/UAS_AI/Container-Detector-Use-KNN/titik_tengah3.json'
with open(output_path, 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"JSON file with combined data saved to {output_path}")
