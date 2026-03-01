import os
import csv
import numpy as np
from PIL import Image
from skimage import color
from scipy.spatial import cKDTree

# =========================
# KONFIGURACJA
# =========================

INPUT_DIR = "maps_raw"
OUTPUT_DIR = "maps_indexed_v3"
PALETTE_LAB_PATH = "palette/palette_v3_127_lab.npy"
PALETTE_RGB_PATH = "palette/palette_v3_127_rgb.npy"

SNAP_THRESHOLD = 5.0
BLOCK_SIZE = 500_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Wczytanie palety
# =========================

palette_lab = np.load(PALETTE_LAB_PATH)
palette_rgb = np.load(PALETTE_RGB_PATH)

ref_lab = palette_lab[1:13]
ref_indices = np.arange(1, 13)

cluster_lab = palette_lab[13:]
cluster_indices = np.arange(13, 128)

tree = cKDTree(cluster_lab)

# =========================
# Raport zbiorczy
# =========================

report_rows = []

tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".tif")]

print(f"Znaleziono {len(tiff_files)} plików TIFF\n")

for filename in tiff_files:

    print(f"\n===== Przetwarzanie: {filename} =====")

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    img = Image.open(input_path).convert("RGB")
    orig_rgb = np.array(img)

    h, w, _ = orig_rgb.shape
    pixels = orig_rgb.reshape(-1, 3)

    pixels_lab = color.rgb2lab(pixels.reshape(1, -1, 3) / 255.0)[0]

    indices = np.zeros(len(pixels_lab), dtype=np.uint8)

    # --- MAPOWANIE ---
    for start in range(0, len(pixels_lab), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, len(pixels_lab))
        block = pixels_lab[start:end]

        # SNAP do referencyjnych
        delta_ref = np.sqrt(
            np.sum((block[:, None, :] - ref_lab[None, :, :]) ** 2, axis=2)
        )

        min_ref_dist = np.min(delta_ref, axis=1)
        min_ref_idx = np.argmin(delta_ref, axis=1)

        snap_mask = min_ref_dist < SNAP_THRESHOLD

        indices[start:end][snap_mask] = ref_indices[min_ref_idx[snap_mask]]

        # KDTree dla reszty
        remaining_mask = ~snap_mask
        if np.any(remaining_mask):
            remaining_block = block[remaining_mask]
            _, idx = tree.query(remaining_block, k=1)
            indices[start:end][remaining_mask] = cluster_indices[idx]

    # --- ZAPIS TIFF INDEXED ---
    indexed_img = Image.fromarray(indices.reshape(h, w), mode="P")

    full_palette = np.zeros((256, 3), dtype=np.uint8)
    full_palette[:128] = palette_rgb
    indexed_img.putpalette(full_palette.flatten().tolist())

    indexed_img.save(output_path, compression="tiff_lzw")

    # --- WALIDACJA ---
    conv_rgb = np.array(Image.open(output_path).convert("RGB"))

    orig_lab = color.rgb2lab(orig_rgb / 255.0)
    conv_lab = color.rgb2lab(conv_rgb / 255.0)

    delta_e = np.sqrt(np.sum((orig_lab - conv_lab) ** 2, axis=2))

    mean_de = float(np.mean(delta_e))
    max_de = float(np.max(delta_e))
    pct_gt5 = float(np.sum(delta_e > 5) / (h*w) * 100)
    pct_gt10 = float(np.sum(delta_e > 10) / (h*w) * 100)

    print(f"Średni ΔE: {mean_de:.3f}")
    print(f"% ΔE > 5: {pct_gt5:.3f}%")
    print(f"% ΔE > 10: {pct_gt10:.3f}%")

    report_rows.append([
        filename,
        mean_de,
        max_de,
        pct_gt5,
        pct_gt10
    ])

# =========================
# Zapis raportu CSV
# =========================

csv_path = os.path.join(OUTPUT_DIR, "batch_report.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "mean_deltaE",
        "max_deltaE",
        "pct_deltaE_gt5",
        "pct_deltaE_gt10"
    ])
    writer.writerows(report_rows)

print("\n===== BATCH ZAKOŃCZONY =====")
print(f"Raport zapisany do: {csv_path}")
