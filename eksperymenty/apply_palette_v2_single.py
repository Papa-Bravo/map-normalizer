import os
import numpy as np
from PIL import Image
from skimage import color
from scipy.spatial import cKDTree

# =========================
# KONFIGURACJA
# =========================

INPUT_FILE = "maps_raw/1020_11.tif"   # <-- zmień na testową mapę
OUTPUT_DIR = "maps_indexed_v2"
PALETTE_LAB_PATH = "palette/palette_v2_lab.npy"
PALETTE_RGB_PATH = "palette/palette_v2_rgb.npy"

BLOCK_SIZE = 500_000  # liczba pikseli przetwarzanych jednorazowo

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1️⃣ Wczytanie palety
# =========================

print("Wczytywanie palety...")

palette_lab = np.load(PALETTE_LAB_PATH)
palette_rgb = np.load(PALETTE_RGB_PATH)

tree = cKDTree(palette_lab)

# =========================
# 2️⃣ Wczytanie obrazu
# =========================

print("Wczytywanie obrazu...")

img = Image.open(INPUT_FILE).convert("RGB")
arr = np.array(img)

h, w, _ = arr.shape
pixels = arr.reshape(-1, 3)

print(f"Rozmiar: {w}x{h}")
print(f"Liczba pikseli: {len(pixels):,}")

# =========================
# 3️⃣ Konwersja RGB → LAB
# =========================

print("Konwersja do LAB...")

pixels_lab = color.rgb2lab(pixels.reshape(1, -1, 3) / 255.0)[0]

# =========================
# 4️⃣ Nearest Neighbor (blokowo)
# =========================

print("Mapowanie do palety...")

indices = np.zeros(len(pixels_lab), dtype=np.uint8)

for start in range(0, len(pixels_lab), BLOCK_SIZE):
    end = min(start + BLOCK_SIZE, len(pixels_lab))
    block = pixels_lab[start:end]

    dist, idx = tree.query(block, k=1)
    indices[start:end] = idx.astype(np.uint8)

print("Mapowanie zakończone.")

# =========================
# 5️⃣ Budowa obrazu indexed
# =========================

indexed_img = Image.fromarray(indices.reshape(h, w), mode="P")

# Pillow wymaga palety 768 elementów (256 * 3)
flat_palette = palette_rgb.flatten().tolist()
indexed_img.putpalette(flat_palette)

# =========================
# 6️⃣ Zapis TIFF Indexed + LZW
# =========================

output_path = os.path.join(
    OUTPUT_DIR,
    os.path.basename(INPUT_FILE)
)

indexed_img.save(
    output_path,
    compression="tiff_lzw"
)

print(f"Zapisano: {output_path}")
