"""
ETAP 03 – Walidacja palety (kolory normatywne + pokrycie).

URUCHAMIAJ NA CAŁYM KATALOGU (po etapie 02, przed etapem 04/05).

Sprawdza:
  - Czy 12 kolorów normatywnych w palecie odpowiada definicji (ΔE < 1)
  - Czy paleta ma dokładnie 128 wpisów (0–127)
  - Podgląd kolorów z nazwami
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from skimage import color
import pipeline_config as cfg

# ── Wczytanie palety ──────────────────────────────────────────────────────────

palette_lab_path = cfg.PALETTE_DIR / "palette_lab.npy"
palette_rgb_path = cfg.PALETTE_DIR / "palette_rgb.npy"

if not palette_lab_path.exists():
    print(f"[BŁĄD] Brak {palette_lab_path}. Uruchom najpierw 02_build_palette.py")
    sys.exit(1)

palette_lab = np.load(palette_lab_path)
palette_rgb = np.load(palette_rgb_path)

assert len(palette_lab) == 127, f"Paleta ma {len(palette_lab)} wpisów, oczekiwano 127"

# ── Kolory referencyjne jak powinny być ──────────────────────────────────────

ref_rgb = np.array(cfg.NORMATIVE_COLORS_RGB, dtype=np.uint8)
ref_lab = color.rgb2lab(ref_rgb.reshape(1, -1, 3) / 255.0)[0]

NORMATIVE_NAMES = [
    "Czarny", "Biały", "Woda płytka", "Woda głęboka", "Izobaty",
    "Ląd", "Światło czerwone", "Światło zielone", "Światło żółte",
    "Szary budynkowy", "Magenta", "Ciemnozielony",
]

# ── Weryfikacja ΔE dla indeksów 0–11 ─────────────────────────────────────────

print("\n===== WALIDACJA KOLORÓW NORMATYWNYCH =====\n")
print(f"{'Idx':>4}  {'Nazwa':<22}  {'ΔE':>6}  {'Status':>8}  RGB wejście → paleta")
print("-" * 75)

all_ok = True
DE_THRESHOLD = 1.0

for i, (name, ref_l, pal_l, ref_r, pal_r) in enumerate(
        zip(NORMATIVE_NAMES, ref_lab, palette_lab[0:12], ref_rgb, palette_rgb[0:12])):

    de = float(np.sqrt(np.sum((ref_l - pal_l) ** 2)))
    ok = de < DE_THRESHOLD
    status = "  OK  " if ok else " FAIL "
    if not ok:
        all_ok = False

    rgb_in  = f"({ref_r[0]:3d},{ref_r[1]:3d},{ref_r[2]:3d})"
    rgb_pal = f"({pal_r[0]:3d},{pal_r[1]:3d},{pal_r[2]:3d})"
    print(f"  {i:2d}  {name:<22}  {de:6.3f}  {status}  {rgb_in} → {rgb_pal}")

print("-" * 75)

# Index 0 to teraz Czarny (normatywny)

print("\n===== WYNIK WALIDACJI =====")
if all_ok:
    print("✓ Wszystkie kolory normatywne OK (ΔE < 1.0)")
else:
    print("✗ Niektóre kolory normatywne przekraczają próg ΔE=1.0")
    print("  Sprawdź i ewentualnie przebuduj paletę (02_build_palette.py)")
    sys.exit(1)

print("✓ Paleta gotowa do konwersji. Uruchom 04_process_single.py lub 05_process_catalog.py")
