import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import color
from PIL import Image

# =========================
# KONFIGURACJA
# =========================

ANALYSIS_PATH = "analysis/sample_lab.npy"
PALETTE_DIR = "palette"

TOTAL_COLORS = 127
RESERVED_INDEX = 0
REF_COUNT = 12

# alokacja segmentów (razem = 115)
SEGMENTS = {
    "very_bright": {"range": (90, 101), "k": 35},
    "bright": {"range": (70, 90), "k": 30},
    "mid": {"range": (40, 70), "k": 25},
    "dark": {"range": (0, 40), "k": 25},
}

RANDOM_STATE = 42
BATCH_SIZE = 10000
MAX_ITER = 200

os.makedirs(PALETTE_DIR, exist_ok=True)

# =========================
# 1️⃣ Kolory referencyjne (kolejność = indeksy 1–12)
# =========================

reference_colors_rgb = [
    (0, 0, 0),           # 1 Czarny
    (255, 255, 255),     # 2 Biały
    (150, 200, 240),     # 3 Płytko
    (200, 230, 250),     # 4 Głębiej
    (0, 90, 170),        # 5 Izobaty
    (242, 234, 199),     # 6 Ląd
    (220, 0, 0),         # 7 Czerwony światłowy
    (0, 160, 0),         # 8 Zielony światłowy
    (255, 220, 0),       # 9 Żółty światłowy
    (160, 160, 160),     # 10 Szary budynkowy
    (200, 0, 120),       # 11 Magenta
    (80, 140, 80),       # 12 Ciemnozielony osuchowy
]

ref_rgb = np.array(reference_colors_rgb)
ref_lab = color.rgb2lab(ref_rgb.reshape(1, -1, 3) / 255.0)[0]

# =========================
# 2️⃣ Wczytanie próbek
# =========================

print("Wczytywanie próbek LAB...")
samples_lab = np.load(ANALYSIS_PATH)

# =========================
# 3️⃣ Usunięcie pikseli bliskich referencyjnym
# =========================

print("Usuwanie pikseli bliskich kolorom referencyjnym...")

DELTA_E_THRESHOLD = 6.0
mask_keep = np.ones(len(samples_lab), dtype=bool)

for ref in ref_lab:
    delta_e = np.sqrt(np.sum((samples_lab - ref) ** 2, axis=1))
    mask_keep &= (delta_e > DELTA_E_THRESHOLD)

filtered_samples = samples_lab[mask_keep]

print(f"Pozostało próbek: {len(filtered_samples):,}")

# =========================
# 4️⃣ Segmentacja i klasteryzacja
# =========================

segment_centroids = []

for name, config in SEGMENTS.items():
    L_min, L_max = config["range"]
    k = config["k"]

    mask = (filtered_samples[:, 0] >= L_min) & (filtered_samples[:, 0] < L_max)
    segment_data = filtered_samples[mask]

    print(f"\nSegment: {name}")
    print(f"  Zakres L*: {L_min}-{L_max}")
    print(f"  Próbki: {len(segment_data):,}")
    print(f"  Klastry: {k}")

    if len(segment_data) < k:
        raise ValueError(f"Za mało próbek w segmencie {name}")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        n_init=5,
        verbose=0
    )

    kmeans.fit(segment_data)
    segment_centroids.append(kmeans.cluster_centers_)

# =========================
# 5️⃣ Składanie palety LAB
# =========================

cluster_lab = np.vstack(segment_centroids)

assert len(cluster_lab) == 115

palette_lab = np.vstack([
    np.zeros((1, 3)),  # indeks 0 RESERVED (dummy)
    ref_lab,           # indeksy 1–12
    cluster_lab        # indeksy 13–127
])

assert len(palette_lab) == 128  # 0..127

# =========================
# 6️⃣ Konwersja LAB → RGB
# =========================

palette_rgb = color.lab2rgb(palette_lab.reshape(1, -1, 3))[0]
palette_rgb = np.clip(palette_rgb, 0, 1)
palette_rgb_uint8 = (palette_rgb * 255).astype(np.uint8)

# indeks 0 = czarny dummy (nieużywany)
palette_rgb_uint8[0] = (0, 0, 0)

# =========================
# 7️⃣ Zapis
# =========================

np.save(os.path.join(PALETTE_DIR, "palette_v3_127_lab.npy"), palette_lab)
np.save(os.path.join(PALETTE_DIR, "palette_v3_127_rgb.npy"), palette_rgb_uint8)

np.savetxt(
    os.path.join(PALETTE_DIR, "palette_v3_127_rgb.csv"),
    palette_rgb_uint8,
    fmt="%d",
    delimiter=",",
    header="R,G,B",
    comments=""
)

# =========================
# 8️⃣ Preview PNG
# =========================

cols = 16
rows = 8
swatch = 40

preview = Image.new("RGB", (cols * swatch, rows * swatch))

for i, rgb in enumerate(palette_rgb_uint8):
    row = i // cols
    col = i % cols
    block = Image.new("RGB", (swatch, swatch), tuple(rgb))
    preview.paste(block, (col * swatch, row * swatch))

preview.save(os.path.join(PALETTE_DIR, "palette_v3_127_preview.png"))

print("\n===== PALETA V3 (127) GOTOWA =====")
print("Indeks 0 = RESERVED")
print("Indeksy 1–12 = referencyjne")
print("Indeksy 13–127 = segmentowane centroidy")
