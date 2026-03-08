import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import color
from PIL import Image

# =========================
# KONFIGURACJA
# =========================

ANALYSIS_DIR = "analysis"
PALETTE_DIR = "palette"

INPUT_SAMPLE = os.path.join(ANALYSIS_DIR, "sample_lab.npy")

K_TOTAL = 256
K_RESERVED = 8
K_CLUSTER = K_TOTAL - K_RESERVED

DELTA_E_THRESHOLD = 6.0  # próg usuwania pikseli bliskich referencyjnym

RANDOM_STATE = 42
BATCH_SIZE = 10_000
MAX_ITER = 200

os.makedirs(PALETTE_DIR, exist_ok=True)

# =========================
# 1️⃣ Kolory referencyjne
# =========================

reference_colors_rgb = {
    "LAND": (242, 234, 199),
    "SHALLOW": (176, 216, 245),
    "DEEP": (255, 255, 255),
    "CONTOUR": (0, 90, 170),
    "BLACK": (0, 0, 0),
    "MAGENTA": (200, 0, 120),
    "SECTOR_RED": (220, 0, 0),
    "SECTOR_GREEN": (0, 160, 0),
}

ref_rgb_array = np.array(list(reference_colors_rgb.values()))
ref_lab = color.rgb2lab(ref_rgb_array.reshape(1, -1, 3) / 255.0)[0]

# =========================
# 2️⃣ Wczytanie próbek
# =========================

print("Wczytywanie próbek LAB...")
samples_lab = np.load(INPUT_SAMPLE)
print(f"Liczba próbek wejściowych: {len(samples_lab):,}")

# =========================
# 3️⃣ Usuwanie pikseli bliskich referencyjnym
# =========================

print("Usuwanie pikseli bliskich kolorom referencyjnym...")

mask_keep = np.ones(len(samples_lab), dtype=bool)

for ref in ref_lab:
    delta_e = np.sqrt(np.sum((samples_lab - ref) ** 2, axis=1))
    mask_keep &= (delta_e > DELTA_E_THRESHOLD)

filtered_samples = samples_lab[mask_keep]

print(f"Po filtracji zostało: {len(filtered_samples):,} próbek")

# =========================
# 4️⃣ MiniBatchKMeans (248 kolorów)
# =========================

print(f"Budowanie {K_CLUSTER} klastrów...")

kmeans = MiniBatchKMeans(
    n_clusters=K_CLUSTER,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    max_iter=MAX_ITER,
    n_init=5,
    verbose=1
)

kmeans.fit(filtered_samples)

centroids_lab = kmeans.cluster_centers_

# =========================
# 5️⃣ Składanie palety
# =========================

# Najpierw referencyjne, potem centroidy
palette_lab = np.vstack([ref_lab, centroids_lab])

# Sortowanie deterministyczne po L*, a*, b*
sort_idx = np.lexsort((
    palette_lab[:, 2],
    palette_lab[:, 1],
    palette_lab[:, 0],
))

palette_lab_sorted = palette_lab[sort_idx]

# =========================
# 6️⃣ Konwersja LAB → RGB
# =========================

palette_rgb = color.lab2rgb(palette_lab_sorted.reshape(1, -1, 3))[0]
palette_rgb = np.clip(palette_rgb, 0, 1)
palette_rgb_uint8 = (palette_rgb * 255).astype(np.uint8)

# =========================
# 7️⃣ Zapis
# =========================

np.save(os.path.join(PALETTE_DIR, "palette_v2_lab.npy"), palette_lab_sorted)
np.save(os.path.join(PALETTE_DIR, "palette_v2_rgb.npy"), palette_rgb_uint8)

np.savetxt(
    os.path.join(PALETTE_DIR, "palette_v2_rgb.csv"),
    palette_rgb_uint8,
    fmt="%d",
    delimiter=",",
    header="R,G,B",
    comments=""
)

# =========================
# 8️⃣ Preview PNG
# =========================

swatch_size = 40
cols = 16
rows = K_TOTAL // cols

palette_image = Image.new("RGB", (cols * swatch_size, rows * swatch_size))

for i, color_rgb in enumerate(palette_rgb_uint8):
    row = i // cols
    col = i % cols
    block = Image.new("RGB", (swatch_size, swatch_size), tuple(color_rgb))
    palette_image.paste(block, (col * swatch_size, row * swatch_size))

palette_image.save(os.path.join(PALETTE_DIR, "palette_v2_preview.png"))

# =========================
# 9️⃣ Raport
# =========================

report = f"""
===== PALETTE V2 REPORT =====

Kolory referencyjne: {K_RESERVED}
Kolory z klasteryzacji: {K_CLUSTER}

Próbki wejściowe: {len(samples_lab):,}
Po filtracji: {len(filtered_samples):,}

Zakres L*:
  min={palette_lab_sorted[:,0].min():.2f}
  max={palette_lab_sorted[:,0].max():.2f}

=================================
"""

print(report)

with open(os.path.join(PALETTE_DIR, "palette_v2_report.txt"), "w") as f:
    f.write(report)

print("Paleta v2 zapisana w katalogu 'palette/'")
