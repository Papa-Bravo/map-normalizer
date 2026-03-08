"""
ETAP 02 – Budowa globalnej palety 127 kolorów (KAP-ready).

URUCHAMIAJ NA CAŁYM KATALOGU (po etapie 01).

Wejście : analysis/sample_lab.npy
          pipeline_config.NORMATIVE_COLORS_RGB
Wyjście : palette/palette_lab.npy   (128×3, indeks 0 = reserved)
          palette/palette_rgb.npy   (128×3 uint8)
          palette/palette_rgb.csv
          palette/palette_preview.png

Struktura palety:
  indeksy 0–11   → kolory normatywne (stałe, semantyczne)
  indeksy 12–126 → centroidy klastrów (wyuczone ze zbioru, razem 115)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
import pipeline_config as cfg

cfg.PALETTE_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Kolory normatywne → LAB ────────────────────────────────────────────────

ref_rgb = np.array(cfg.NORMATIVE_COLORS_RGB, dtype=np.uint8)  # (12, 3)
ref_lab = color.rgb2lab(ref_rgb.reshape(1, -1, 3) / 255.0)[0]  # (12, 3)

print(f"Zdefiniowano {cfg.REF_COUNT} kolorów normatywnych.")

# ── 2. Wczytanie próbek ───────────────────────────────────────────────────────

sample_path = cfg.ANALYSIS_DIR / "sample_lab.npy"
if not sample_path.exists():
    print(f"[BŁĄD] Brak {sample_path}. Uruchom najpierw 01_analyze_catalog.py")
    sys.exit(1)

print("Wczytywanie próbek LAB...")
samples = np.load(sample_path)
print(f"  Próbki: {len(samples):,}")

# ── 3. Wykluczenie próbek bliskich normatywnym ────────────────────────────────
# (te obszary pokryją normatywne indeksy; nie muszą dostawać własnych klastrów)

print(f"Wykluczanie próbek w odległości ΔE < {cfg.DELTA_E_REF_EXCL} od normatywnych...")
keep = np.ones(len(samples), dtype=bool)

for ref in ref_lab:
    delta_e = np.sqrt(np.sum((samples - ref) ** 2, axis=1))
    keep &= (delta_e > cfg.DELTA_E_REF_EXCL)

filtered = samples[keep]
print(f"  Pozostało: {len(filtered):,} próbek ({100*len(filtered)/len(samples):.1f}%)")

# ── 4. Segmentacja i klasteryzacja per segment L* ─────────────────────────────

expected_detail = cfg.TOTAL_COLORS - cfg.REF_COUNT  # 127 - 12 = 115
k_total = sum(s["k"] for s in cfg.PALETTE_SEGMENTS.values())
assert k_total == expected_detail, (
    f"Suma k w PALETTE_SEGMENTS ({k_total}) musi równać się {expected_detail}"
)

centroids = []

if not cfg.USE_SEGMENTED_PALETTE:
    print(f"\nUruchamiam GLOBALNY KMeans dla {expected_detail} klastrów...")
    # Globalny KMeans na wszystkich przefiltrowanych próbkach
    kmeans = KMeans(
        n_clusters=expected_detail,
        random_state=cfg.KMEANS_SEED,
        max_iter=cfg.KMEANS_ITER,
        n_init=10,
    )
    # Dla wydajności przy globalnym KMeans, jeśli próbek jest za dużo, wybieramy podzbiór
    MAX_PALETTE_SAMPLES = 1_000_000
    if len(filtered) > MAX_PALETTE_SAMPLES:
        print(f"  Ograniczam próbkę do {MAX_PALETTE_SAMPLES:,} dla stabilności KMeans...")
        sub_idx = np.random.choice(len(filtered), MAX_PALETTE_SAMPLES, replace=False)
        kmeans.fit(filtered[sub_idx])
    else:
        kmeans.fit(filtered)
    detail_lab = kmeans.cluster_centers_
else:
    centroids = []
    for name, seg in cfg.PALETTE_SEGMENTS.items():
        L_min, L_max = seg["range"]
        k = seg["k"]
        mask = (filtered[:, 0] >= L_min) & (filtered[:, 0] < L_max)
        seg_data = filtered[mask]
        print(f"\nSegment '{name}'  L*=[{L_min},{L_max})  próbek={len(seg_data):,}  k={k}")
        if len(seg_data) < k:
            raise ValueError(f"Za mało próbek w segmencie '{name}'")
        kmeans = KMeans(
            n_clusters=k,
            random_state=cfg.KMEANS_SEED,
            max_iter=cfg.KMEANS_ITER,
            n_init=10,
        )
        kmeans.fit(seg_data)
        centroids.append(kmeans.cluster_centers_)
    detail_lab = np.vstack(centroids)
assert len(detail_lab) == expected_detail

# ── 5. Składanie palety ───────────────────────────────────────────────────────

palette_lab = np.vstack([
    ref_lab,            # indeksy 0–11: normatywne
    detail_lab,         # indeksy 12–126: klastry detaliczne
])
assert len(palette_lab) == 127, f"Oczekiwano 127 wpisów, mam {len(palette_lab)}"

# ── 6. Konwersja LAB → RGB ────────────────────────────────────────────────────

palette_rgb_f = color.lab2rgb(palette_lab.reshape(1, -1, 3))[0]
palette_rgb_f = np.clip(palette_rgb_f, 0.0, 1.0)
palette_rgb = (palette_rgb_f * 255).astype(np.uint8)
# Brak placeholderów - indeks 0 to pierwszy kolor normatywny (Czarny)

# ── 7. Zapis ──────────────────────────────────────────────────────────────────

np.save(cfg.PALETTE_DIR / "palette_lab.npy", palette_lab)
np.save(cfg.PALETTE_DIR / "palette_rgb.npy", palette_rgb)

np.savetxt(
    cfg.PALETTE_DIR / "palette_rgb.csv",
    palette_rgb, fmt="%d", delimiter=",",
    header="R,G,B", comments="",
)

# ── 8. Preview PNG ────────────────────────────────────────────────────────────

COLS, ROWS, CELL = 16, 8, 40
preview = Image.new("RGB", (COLS * CELL, ROWS * CELL))

for i, rgb in enumerate(palette_rgb):
    row, col = divmod(i, COLS)
    preview.paste(Image.new("RGB", (CELL, CELL), tuple(rgb)), (col * CELL, row * CELL))

preview.save(cfg.PALETTE_DIR / "palette_preview.png")

print(f"\n===== PALETA GOTOWA =====")
print(f"  Łącznie kolorów: {len(palette_lab)}")
print(f"  Indeksy 0–11  : normatywne")
print(f"  Indeksy 12–126: klastry detaliczne")
print(f"  Zapisano do {cfg.PALETTE_DIR}")
print("✓ Etap 02 zakończony. Uruchom 03_validate_palette.py")
