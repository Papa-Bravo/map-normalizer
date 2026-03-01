import os
import csv
import numpy as np
from PIL import Image

INPUT_DIR = "maps_indexed_v3"
OUTPUT_CSV = os.path.join(INPUT_DIR, "palette_usage_report.csv")

tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".tif")]

print(f"Znaleziono {len(tiff_files)} plików TIFF\n")

rows = []

for filename in tiff_files:

    path = os.path.join(INPUT_DIR, filename)
    img = Image.open(path)
    indices = np.array(img)

    flat = indices.flatten()
    hist = np.bincount(flat, minlength=128)

    total_pixels = len(flat)
    used_indices = np.sum(hist > 0)
    unused_indices = np.sum(hist == 0)
    index0_used = hist[0]

    used_real_colors = used_indices - (1 if hist[0] > 0 else 0)
    utilization_pct = (used_real_colors / 127) * 100

    rows.append([
        filename,
        total_pixels,
        used_indices,
        unused_indices,
        index0_used,
        used_real_colors,
        utilization_pct
    ])

    print(f"{filename}")
    print(f"  Użyte indeksy: {used_indices}")
    print(f"  Indeks 0 użyty: {index0_used}")
    print(f"  Realne kolory użyte: {used_real_colors}/127 "
          f"({utilization_pct:.2f}%)\n")

# --- zapis CSV ---

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "total_pixels",
        "used_indices_0_127",
        "unused_indices",
        "index0_pixel_count",
        "used_real_colors_1_127",
        "utilization_percent"
    ])
    writer.writerows(rows)

print("===== BATCH WERYFIKACJI ZAKOŃCZONY =====")
print(f"Raport zapisany do: {OUTPUT_CSV}")
