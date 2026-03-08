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

K = 256
RANDOM_STATE = 42
BATCH_SIZE = 10_000
MAX_ITER = 200

os.makedirs(PALETTE_DIR, exist_ok=True)

# =========================
# 1️⃣ Wczytanie próbek LAB
# =========================

print("Wczytywanie próbek LAB...")
samples_lab = np.load(INPUT_SAMPLE)

print(f"Liczba próbek: {len(samples_lab):,}")

# =========================
# 2️⃣ MiniBatchKMeans
# =========================

print("Budowanie klastrów (MiniBatchKMeans)...")

kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    max_iter=MAX_ITER,
    n_init=5,
    verbose=1
)

kmeans.fit(samples_lab)

centroids_lab = kmeans.cluster_centers_

# =========================
# 3️⃣ Sortowanie centroidów
# =========================
# Sortujemy deterministycznie:
# najpierw po L*, potem a*, potem b*

sort_idx = np.lexsort((
    centroids_lab[:, 2],  # b*
    centroids_lab[:, 1],  # a*
    centroids_lab[:, 0],  # L*
))

centroids_lab_sorted = centroids_lab[sort_idx]

# =========================
# 4️⃣ Konwersja LAB → RGB
# =========================

centroids_lab_reshaped = centroids_lab_sorted.reshape(1, -1, 3)
centroids_rgb = color.lab2rgb(centroids_lab_reshaped)[0]

# clamp (na wszelki wypadek)
centroids_rgb = np.clip(centroids_rgb, 0, 1)

centroids_rgb_uint8 = (centroids_rgb * 255).astype(np.uint8)

# =========================
# 5️⃣ Zapis palety
# =========================

np.save(os.path.join(PALETTE_DIR, "palette_v1_lab.npy"), centroids_lab_sorted)
np.save(os.path.join(PALETTE_DIR, "palette_v1_rgb.npy"), centroids_rgb_uint8)

# zapis CSV dla czytelności
np.savetxt(
    os.path.join(PALETTE_DIR, "palette_v1_rgb.csv"),
    centroids_rgb_uint8,
    fmt="%d",
    delimiter=",",
    header="R,G,B",
    comments=""
)

# =========================
# 6️⃣ Wizualizacja PNG
# =========================

swatch_size = 40
cols = 16
rows = K // cols

palette_image = Image.new("RGB", (cols * swatch_size, rows * swatch_size))

for i, color_rgb in enumerate(centroids_rgb_uint8):
    row = i // cols
    col = i % cols

    block = Image.new("RGB", (swatch_size, swatch_size), tuple(color_rgb))
    palette_image.paste(block, (col * swatch_size, row * swatch_size))

palette_image.save(os.path.join(PALETTE_DIR, "palette_v1_preview.png"))

# =========================
# 7️⃣ Raport
# =========================

report = f"""
===== PALETTE BUILD REPORT =====

K (liczba kolorów): {K}
Liczba próbek wejściowych: {len(samples_lab):,}

Zakres L*:
  min={centroids_lab_sorted[:,0].min():.2f}
  max={centroids_lab_sorted[:,0].max():.2f}

Zakres a*:
  min={centroids_lab_sorted[:,1].min():.2f}
  max={centroids_lab_sorted[:,1].max():.2f}

Zakres b*:
  min={centroids_lab_sorted[:,2].min():.2f}
  max={centroids_lab_sorted[:,2].max():.2f}

=================================
"""

print(report)

with open(os.path.join(PALETTE_DIR, "palette_v1_report.txt"), "w") as f:
    f.write(report)

print("Paleta zapisana w katalogu 'palette/'")
