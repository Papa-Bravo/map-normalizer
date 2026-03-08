"""
ETAP 01 – Analiza kolorów całego katalogu map.

URUCHAMIAJ NA CAŁYM KATALOGU (przed budową palety).

Wejście : INPUT_DIR (TIFF + BMP, zdefiniowane w pipeline_config.py)
Wyjście : analysis/sample_lab.npy
          analysis/report.txt

Co robi:
  1. Wczytuje każdy plik TIFF/BMP z INPUT_DIR
  2. Usuwa piksele czystej bieli (ramki skanera)
  3. Próbkuje losowo 3% pikseli
  4. Konwertuje RGB → CIELAB
  5. Redukuje nadreprezentację bardzo jasnych tonów (woda, niebo)
  6. Łączy i zapisuje globalną próbkę LAB
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from tqdm import tqdm
import pipeline_config as cfg

# ── Inicjalizacja ─────────────────────────────────────────────────────────────

np.random.seed(cfg.RANDOM_SEED)
cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Zbieranie plików ──────────────────────────────────────────────────────────

input_files = [
    p for p in sorted(cfg.INPUT_DIR.iterdir())
    if p.suffix.lower() in cfg.SUPPORTED_EXTENSIONS
]

if not input_files:
    print(f"[BŁĄD] Brak plików w {cfg.INPUT_DIR}")
    print(f"       Obsługiwane formaty: {cfg.SUPPORTED_EXTENSIONS}")
    sys.exit(1)

print(f"Znaleziono {len(input_files)} plików w {cfg.INPUT_DIR}")

# ── Próbkowanie ───────────────────────────────────────────────────────────────

all_samples = []
total_pixels = 0
total_white_rejected = 0

for path in tqdm(input_files, desc="Analiza"):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    num_pixels = h * w
    total_pixels += num_pixels

    pixels = arr.reshape(-1, 3)

    # 1) Usuń czystą biel (ramki skanera)
    white_mask = (
        (pixels[:, 0] > cfg.WHITE_THRESHOLD) &
        (pixels[:, 1] > cfg.WHITE_THRESHOLD) &
        (pixels[:, 2] > cfg.WHITE_THRESHOLD)
    )
    total_white_rejected += int(white_mask.sum())
    pixels = pixels[~white_mask]

    if len(pixels) == 0:
        continue

    # 2) Losowe próbkowanie
    n_sample = min(int(len(pixels) * cfg.SAMPLE_PERCENT), cfg.MAX_TOTAL_SAMPLES)
    if n_sample == 0:
        continue

    idx = np.random.choice(len(pixels), n_sample, replace=False)
    sampled = pixels[idx]

    # 3) RGB → CIELAB
    lab = color.rgb2lab(sampled.reshape(1, -1, 3) / 255.0)[0]

    # 4) Redukcja nadreprezentacji bardzo jasnych tonów
    L = lab[:, 0]
    bright_idx = np.where(L > cfg.BRIGHT_L_THRESHOLD)[0]
    if len(bright_idx) > 0:
        n_drop = int(len(bright_idx) * (1.0 - cfg.BRIGHT_SAMPLING_RATIO))
        drop = np.random.choice(bright_idx, n_drop, replace=False)
        keep = np.ones(len(lab), dtype=bool)
        keep[drop] = False
        lab = lab[keep]

    all_samples.append(lab)

# ── Łączenie i zapis ──────────────────────────────────────────────────────────

if not all_samples:
    print("[BŁĄD] Brak próbek – sprawdź dane wejściowe.")
    sys.exit(1)

samples = np.vstack(all_samples)

L = samples[:, 0]
a = samples[:, 1]
b = samples[:, 2]

report = f"""===== RAPORT ANALIZY KATALOGU =====

Liczba plików      : {len(input_files)}
Łączna l. pikseli  : {total_pixels:,}
Odrzucone (biel)   : {total_white_rejected:,} ({100 * total_white_rejected / total_pixels:.1f}%)
Próbka LAB         : {len(samples):,}

Zakres L* : min={L.min():.2f}  max={L.max():.2f}  mean={L.mean():.2f}
Zakres a* : min={a.min():.2f}  max={a.max():.2f}  mean={a.mean():.2f}
Zakres b* : min={b.min():.2f}  max={b.max():.2f}  mean={b.mean():.2f}

===========================================
"""

print(report)

np.save(cfg.ANALYSIS_DIR / "sample_lab.npy", samples)
(cfg.ANALYSIS_DIR / "report.txt").write_text(report, encoding="utf-8")

print(f"Zapisano: {cfg.ANALYSIS_DIR / 'sample_lab.npy'}")
print(f"Zapisano: {cfg.ANALYSIS_DIR / 'report.txt'}")
print("✓ Etap 01 zakończony. Uruchom 02_build_palette.py")
