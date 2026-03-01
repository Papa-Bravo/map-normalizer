import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import color

# =========================
# KONFIGURACJA
# =========================

INPUT_DIR = "maps_raw"
ANALYSIS_DIR = "analysis"
SAMPLE_PERCENT = 0.03        # 3% pikseli
MAX_TOTAL_SAMPLES = 4_000_000
WHITE_THRESHOLD = 250        # próg czystej bieli RGB
BRIGHT_L_THRESHOLD = 95      # L* powyżej którego redukujemy sampling
BRIGHT_SAMPLING_RATIO = 0.3  # tylko 30% bardzo jasnych pikseli
RANDOM_SEED = 42

# =========================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(ANALYSIS_DIR, exist_ok=True)

all_samples = []
total_pixels = 0
total_rejected_white = 0

tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".tif")]

print(f"Znaleziono {len(tiff_files)} plików TIFF")

for filename in tqdm(tiff_files):
    path = os.path.join(INPUT_DIR, filename)

    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    h, w, _ = arr.shape
    num_pixels = h * w
    total_pixels += num_pixels

    pixels = arr.reshape(-1, 3)

    # 1️⃣ usuwamy czystą biel (ramka skanu)
    mask_not_white = ~(
        (pixels[:, 0] > WHITE_THRESHOLD) &
        (pixels[:, 1] > WHITE_THRESHOLD) &
        (pixels[:, 2] > WHITE_THRESHOLD)
    )

    rejected_white = np.sum(~mask_not_white)
    total_rejected_white += rejected_white

    filtered_pixels = pixels[mask_not_white]

    # 2️⃣ losowe próbkowanie
    sample_size = int(len(filtered_pixels) * SAMPLE_PERCENT)
    sample_size = min(sample_size, MAX_TOTAL_SAMPLES)

    if sample_size == 0:
        continue

    indices = np.random.choice(len(filtered_pixels), sample_size, replace=False)
    sampled = filtered_pixels[indices]

    # 3️⃣ konwersja do LAB
    sampled_rgb_norm = sampled / 255.0
    sampled_lab = color.rgb2lab(sampled_rgb_norm)

    # 4️⃣ ograniczenie dominacji bardzo jasnych
    L_channel = sampled_lab[:, 0]
    bright_mask = L_channel > BRIGHT_L_THRESHOLD

    keep_mask = np.ones(len(sampled_lab), dtype=bool)

    bright_indices = np.where(bright_mask)[0]
    reduce_count = int(len(bright_indices) * (1 - BRIGHT_SAMPLING_RATIO))

    if reduce_count > 0:
        drop_indices = np.random.choice(bright_indices, reduce_count, replace=False)
        keep_mask[drop_indices] = False

    sampled_lab_balanced = sampled_lab[keep_mask]

    all_samples.append(sampled_lab_balanced)

# =========================
# Łączenie próbek
# =========================

if not all_samples:
    raise RuntimeError("Brak próbek – sprawdź dane wejściowe.")

all_samples = np.vstack(all_samples)

# =========================
# Statystyki
# =========================

L_vals = all_samples[:, 0]
a_vals = all_samples[:, 1]
b_vals = all_samples[:, 2]

report = f"""
===== RAPORT ANALIZY =====

Liczba plików: {len(tiff_files)}
Łączna liczba pikseli: {total_pixels:,}
Odrzucone jako czysta biel: {total_rejected_white:,}

Liczba próbek LAB: {len(all_samples):,}

Zakres L*: min={L_vals.min():.2f}, max={L_vals.max():.2f}, mean={L_vals.mean():.2f}
Zakres a*: min={a_vals.min():.2f}, max={a_vals.max():.2f}, mean={a_vals.mean():.2f}
Zakres b*: min={b_vals.min():.2f}, max={b_vals.max():.2f}, mean={b_vals.mean():.2f}

==========================
"""

print(report)

# zapis próbki do dalszego etapu
np.save(os.path.join(ANALYSIS_DIR, "sample_lab.npy"), all_samples)

with open(os.path.join(ANALYSIS_DIR, "report.txt"), "w") as f:
    f.write(report)

print("Zapisano próbkę do analysis/sample_lab.npy")
print("Zapisano raport do analysis/report.txt")
