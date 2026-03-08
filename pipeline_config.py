"""
pipeline_config.py – Centralna konfiguracja pipeline'u standaryzacji kolorów map morskich.

Edytuj TYLKO TEN PLIK przed uruchomieniem kolejnych kroków.
Nie zmieniaj parametrów wewnątrz poszczególnych skryptów.
"""

from pathlib import Path

# ── Katalogi ──────────────────────────────────────────────────────────────────
# Katalog new/ jest rootem projektu.
# Umieść pliki źródłowe w new/maps_raw/ (TIFF / BMP).

ROOT_DIR     = Path(__file__).parent               # .../map_converter/new/

INPUT_DIR    = ROOT_DIR / "maps_raw"               # źródłowe TIFF / BMP (dodaj ręcznie)
OUTPUT_DIR   = ROOT_DIR / "maps_indexed"           # wynikowe TIFF Indexed
ANALYSIS_DIR = ROOT_DIR / "analysis"               # próbki LAB i raporty analityczne
PALETTE_DIR  = ROOT_DIR / "palette"                # pliki palety

# ── Obsługiwane formaty wejściowe ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".bmp")

# ── ETAP 01 – Analiza kolorów katalogu ───────────────────────────────────────

SAMPLE_PERCENT        = 0.03       # 3% pikseli na plik
MAX_TOTAL_SAMPLES     = 5_000_000  # twardy limit łącznej próbki
WHITE_THRESHOLD       = 250        # piksele (R>W & G>W & B>W) = biała ramka → pomijane
BRIGHT_L_THRESHOLD    = 95         # L* powyżej którego redukujemy reprezentację
BRIGHT_SAMPLING_RATIO = 0.70       # retencja bardzo jasnych tonów (70% - zwiększono dla lepszej detekcji wody)
RANDOM_SEED           = 42

# ── ETAP 02 – Budowa palety ───────────────────────────────────────────────────

TOTAL_COLORS     = 127    # maks. BSB/KAP (7-bit; indeks 0 zarezerwowany)
REF_COUNT        = 12     # kolory normatywne na stałych indeksach 1–12

# Czy używać segmentacji jasności (L*) przy budowie palety?
# Globalny KMeans (False) zwykle daje mniejszy błąd średni (ΔE).
USE_SEGMENTED_PALETTE = False

PALETTE_SEGMENTS = {
    "very_bright": {"range": (90, 101), "k": 35},
    "bright":      {"range": (70, 90),  "k": 30},
    "mid":         {"range": (40, 70),  "k": 25},
    "dark":        {"range": (0,  40),  "k": 25},
}

KMEANS_BATCH  = 10_000
KMEANS_ITER   = 200
KMEANS_SEED   = 42

# Próg ΔE: próbki bliżej niż to od normatywnych są wykluczone z klasteryzacji
# Musi być spójny ze SNAP_THRESHOLD (6.0), aby nie było luki w kolorach.
DELTA_E_REF_EXCL = 6.0

# 12 kolorów normatywnych RGB — indeksy 1–12 w palecie (kolejność = indeks!)
NORMATIVE_COLORS_RGB = [
    (  0,   0,   0),   # 1  Czarny          – kontury, napisy
    (255, 255, 255),   # 2  Biały            – tło, napisy
    (150, 200, 240),   # 3  Woda płytka
    (200, 230, 250),   # 4  Woda głęboka / morze otwarte
    (  0,  90, 170),   # 5  Izobaty / linie głębinowe
    (242, 234, 199),   # 6  Ląd
    (220,   0,   0),   # 7  Światło czerwone
    (  0, 160,   0),   # 8  Światło zielone
    (255, 220,   0),   # 9  Światło żółte
    (160, 160, 160),   # 10 Szary budynkowy
    (200,   0, 120),   # 11 Magenta (strefy VTS, ograniczenia)
    ( 80, 140,  80),   # 12 Ciemnozielony (osuchy, roślinność)
]

# ── ETAP 04 – Przetwarzanie pojedynczego pliku ────────────────────────────────

# Korekcja przesunięcia barwnego skanera (color_shift)
# Wymaga pliku ../color_defs.py z listami `scan_colors` i `colors`
USE_COLOR_SHIFT = False  # Domyślnie wyłączony, dopóki użytkownik nie potwierdzi kalibracji

# Filtr bilateral (edge-preserving smoothing) przed kwantyzacją kolorów
# Parametry z v10 (sprawdzone)
USE_BILATERAL          = True
BILATERAL_SIGMA_COLOR  = 0.05   
BILATERAL_SIGMA_SPATIAL = 3     

# Mapowanie pikseli do palety
SNAP_THRESHOLD = 6.0      # ΔE: poniżej tego przypisz piksel do normatywnego (v10: 6.0)
BLOCK_SIZE     = 500_000  # pikseli na blok (ograniczenie zużycia RAM)

# Majority filter na tle (indeksy 13–127) – redukcja pojedynczych artefaktów
USE_MAJORITY_FILTER = True
MAJORITY_KERNEL     = 3    # rozmiar okna (3×3)
MAJORITY_MIN_VOTES  = 6    # min. sąsiadów z dominującym indeksem

# ── ETAP 06/07 – Walidacja jakości (Priorytetyzacja BSB/KAP) ───────────────────

# Kategoria: STRUKTURALNA (CRITICAL - bezpieczeństwo nawigacyjne)
MIN_SSIM          = 0.90   # SSIM (0–1) - ogólna czytelność
MIN_EDGE_IOU      = 0.80   # Pokrycie krawędzi (detale, napisy)
MIN_GRAD_CORR     = 0.90   # Wierność przejść i symboli
MAX_EDGE_DIST_PX  = 1.0    # Maks. przesunięcie krawędzi [px]

# Kategoria: KOLORYSTYCZNA (WARNING - spójność wizualna katalogu)
MAX_MEAN_DE       = 2.0    # Średni błąd ΔE (jeśli > to ostrzeżenie)
MAX_PCT_GT10      = 0.2    # Maks. % dużych błędów (jeśli > to ostrzeżenie)

# Kategoria: NORMATYWNA (CRITICAL - semantyka kolorów)
# Błąd kolorów typu "Czerwone światło" lub "Izobata" musi być minimalny.
MAX_NORMATIVE_DE  = 1.0
