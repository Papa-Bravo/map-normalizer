import os
import numpy as np
from PIL import Image
from skimage import color
from scipy.spatial import cKDTree

# =========================
# KONFIGURACJA
# =========================

INPUT_FILE = "maps_raw/1020_00.tif"   # <-- zmień jeśli potrzeba
OUTPUT_DIR = "maps_indexed_v3"
PALETTE_LAB_PATH = "palette/palette_v3_127_lab.npy"
PALETTE_RGB_PATH = "palette/palette_v3_127_rgb.npy"

SNAP_THRESHOLD = 5.0
BLOCK_SIZE = 500_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1️⃣ Wczytanie palety
# =========================

palette_lab = np.load(PALETTE_LAB_PATH)
palette_rgb = np.load(PALETTE_RGB_PATH)

# referencyjne indeksy 1–12
ref_lab = palette_lab[1:13]
ref_indices = np.arange(1, 13)

# KDTree dla 13–127
cluster_lab = palette_lab[13:]
cluster_indices = np.arange(13, 128)

tree = cKDTree(cluster_lab)

# =========================
# 2️⃣ Wczytanie obrazu
# =========================

img = Image.open(INPUT_FILE).convert("RGB")
orig_rgb = np.array(img)

h, w, _ = orig_rgb.shape
pixels = orig_rgb.reshape(-1, 3)

print(f"Rozmiar: {w}x{h}")
print(f"Piksele: {len(pixels):,}")

# =========================
# 3️⃣ RGB → LAB
# =========================

pixels_lab = color.rgb2lab(pixels.reshape(1, -1, 3) / 255.0)[0]

# =========================
# 4️⃣ Mapowanie
# =========================

indices = np.zeros(len(pixels_lab), dtype=np.uint8)

for start in range(0, len(pixels_lab), BLOCK_SIZE):
    end = min(start + BLOCK_SIZE, len(pixels_lab))
    block = pixels_lab[start:end]

    # ---- SNAP do referencyjnych ----
    delta_ref = np.sqrt(
        np.sum(
            (block[:, None, :] - ref_lab[None, :, :]) ** 2,
            axis=2
        )
    )

    min_ref_dist = np.min(delta_ref, axis=1)
    min_ref_idx = np.argmin(delta_ref, axis=1)

    snap_mask = min_ref_dist < SNAP_THRESHOLD

    # przypisanie snap
    indices[start:end][snap_mask] = ref_indices[min_ref_idx[snap_mask]]

    # ---- KDTree dla pozostałych ----
    remaining_mask = ~snap_mask

    if np.any(remaining_mask):
        remaining_block = block[remaining_mask]
        dist, idx = tree.query(remaining_block, k=1)
        indices[start:end][remaining_mask] = cluster_indices[idx]

print("Mapowanie zakończone.")

# =========================
# 5️⃣ Zapis TIFF Indexed
# =========================

indexed_img = Image.fromarray(indices.reshape(h, w), mode="P")

# Pillow wymaga 256*3 slotów
full_palette = np.zeros((256, 3), dtype=np.uint8)
full_palette[:128] = palette_rgb
indexed_img.putpalette(full_palette.flatten().tolist())

output_path = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE))

indexed_img.save(output_path, compression="tiff_lzw")

print(f"Zapisano: {output_path}")

# =========================
# 6️⃣ WALIDACJA ΔE
# =========================

print("Liczenie jakości...")

conv_rgb = np.array(Image.open(output_path).convert("RGB"))

orig_lab = color.rgb2lab(orig_rgb / 255.0)
conv_lab = color.rgb2lab(conv_rgb / 255.0)

delta_e = np.sqrt(np.sum((orig_lab - conv_lab) ** 2, axis=2))

mean_de = np.mean(delta_e)
max_de = np.max(delta_e)
pct_gt5 = np.sum(delta_e > 5) / (h*w) * 100
pct_gt10 = np.sum(delta_e > 10) / (h*w) * 100

print("\n===== RAPORT WALIDACJI (127 kolorów) =====\n")
print(f"Średni ΔE: {mean_de:.3f}")
print(f"Maksymalny ΔE: {max_de:.3f}")
print(f"% pikseli ΔE > 5:  {pct_gt5:.3f}%")
print(f"% pikseli ΔE > 10: {pct_gt10:.3f}%")
