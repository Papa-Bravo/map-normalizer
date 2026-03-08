"""
ETAP 04 – Przetwarzanie POJEDYNCZEGO pliku mapy.

URUCHAMIAJ PER-PLIK lub użyj 05_process_catalog.py do przetwarzania batch.

Użycie:
    python 04_process_single.py <plik_wejsciowy> [plik_wyjsciowy]

  Jeśli plik_wyjsciowy nie podany → zapisuje do OUTPUT_DIR/<basename>.tif

Pipeline per plik:
  [0] Opcjonalnie: korekcja barwna skanera (color_shift, USE_COLOR_SHIFT=True)
  [1] Opcjonalnie: filtr bilateral (edge-preserving smoothing, USE_BILATERAL=True)
  [2] RGB → CIELAB
  [3] SNAP: piksele ΔE < SNAP_THRESHOLD od normatywnych → stały indeks normatywny
  [4] Pozostałe → cKDTree nearest-neighbor po klastrach 13–127
  [5] Opcjonalnie: majority filter na pikselach tła (USE_MAJORITY_FILTER=True)
  [6] Zapis TIFF Indexed 127 kolorów (LZW)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from skimage.restoration import denoise_bilateral
from scipy.spatial import cKDTree
import pipeline_config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Funkcje pomocnicze
# ══════════════════════════════════════════════════════════════════════════════

def load_palette():
    """Wczytuje paletę LAB i RGB z PALETTE_DIR."""
    lab_path = cfg.PALETTE_DIR / "palette_lab.npy"
    rgb_path = cfg.PALETTE_DIR / "palette_rgb.npy"
    if not lab_path.exists():
        raise FileNotFoundError(
            f"Brak palety: {lab_path}\nUruchom najpierw 02_build_palette.py"
        )
    return np.load(lab_path), np.load(rgb_path)


def apply_color_shift(img_rgb_f: np.ndarray) -> np.ndarray:
    """
    Wektorowa korekcja przesunięcia barwnego skanera (LAB-space offset).

    Dla każdego piksela:
      1. Znajdź najbliższy kolor skanera (scan_color) w przestrzeni LAB
      2. Zastosuj offset = ref_color_LAB - scan_color_LAB

    Wymaga: ../color_defs.py z listami `scan_colors` i `colors` (RGB 0–255 int).
    """
    try:
        parent = str(Path(__file__).parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        import color_defs
    except ImportError:
        print("[UWAGA] Nie znaleziono color_defs.py – pomijam color_shift.")
        return img_rgb_f

    scan_rgb = np.array(color_defs.scan_colors, dtype=np.float32) / 255.0  # (N, 3)
    ref_rgb  = np.array(color_defs.colors,      dtype=np.float32) / 255.0

    offsets = ref_rgb - scan_rgb   # (N, 3) apply additive offset in RGB space

    flat_rgb = img_rgb_f.reshape(-1, 3)
    
    # Znajdź najbliższy kolor skanera w RGB
    tree = cKDTree(scan_rgb)
    _, nn_idx = tree.query(flat_rgb, k=1)

    # Zastosuj offset RGB
    corrected = flat_rgb + offsets[nn_idx]
    
    return np.clip(corrected.reshape(img_rgb_f.shape), 0.0, 1.0).astype(np.float32)


def apply_bilateral(img_rgb_f: np.ndarray) -> np.ndarray:
    """
    Filtr bilateral per kanał (edge-preserving smoothing).
    Wygładza obszary jednolitego koloru BEZ rozmycia krawędzi (napisy, linie).
    """
    smoothed = np.empty_like(img_rgb_f)
    for c in range(3):
        smoothed[:, :, c] = denoise_bilateral(
            img_rgb_f[:, :, c],
            sigma_color=cfg.BILATERAL_SIGMA_COLOR,
            sigma_spatial=cfg.BILATERAL_SIGMA_SPATIAL,
        )
    return smoothed


def map_to_palette(pixels_lab: np.ndarray,
                   ref_lab: np.ndarray,
                   ref_indices: np.ndarray,
                   tree: cKDTree,
                   cluster_indices: np.ndarray) -> np.ndarray:
    """
    Dwuetapowe mapowanie pikseli do indeksów palety.

    Krok A – SNAP: piksele w odległości ΔE < SNAP_THRESHOLD od normatywnych
             dostają stały indeks normatywny (1–12).
    Krok B – Remaining: pozostałe piksele → cKDTree → klaster detaliczny (13–127).

    Przetwarza bloki po BLOCK_SIZE pikseli aby kontrolować zużycie RAM.
    """
    n = len(pixels_lab)
    indices = np.zeros(n, dtype=np.uint8)

    for start in range(0, n, cfg.BLOCK_SIZE):
        end = min(start + cfg.BLOCK_SIZE, n)
        block = pixels_lab[start:end]

        # ── A: SNAP do normatywnych ────────────────────────────────────────
        delta = np.sqrt(
            np.sum((block[:, None, :] - ref_lab[None, :, :]) ** 2, axis=2)
        )  # (B, 12)
        min_dist = delta.min(axis=1)
        min_idx  = delta.argmin(axis=1)

        snap_mask = min_dist < cfg.SNAP_THRESHOLD
        indices[start:end][snap_mask] = ref_indices[min_idx[snap_mask]]

        # ── B: Pozostałe → KDTree ──────────────────────────────────────────
        rest_mask = ~snap_mask
        if rest_mask.any():
            _, nn = tree.query(block[rest_mask], k=1)
            indices[start:end][rest_mask] = cluster_indices[nn]

    return indices


def apply_majority_filter(index_img: np.ndarray) -> np.ndarray:
    """
    Majority (modal) filter tylko dla indeksów tła (13–127).
    Usuwa pojedyncze "zaszumione" piksele w jednolitych obszarach.
    Indeksy normatywne 1–12 są nigdy nie dotykane.

    Używa skimage.filters.rank.modal (C-implementation, szybkie).
    """
    # Implementacja z progiem decyzyjnym (v9 style)
    # Jeśli dominujący sąsiad ma >= MIN_VOTES, wybierz go. W przeciwnym razie zostaw oryginał.
    # Używamy scipy.ndimage.generic_filter dla precyzyjnej kontroli progu.
    from scipy.ndimage import generic_filter
    
    def _majority_with_threshold(vals):
        center = vals[len(vals)//2]
        counts = np.bincount(vals.astype(np.uint8), minlength=256)
        dominant = np.argmax(counts)
        if counts[dominant] >= cfg.MAJORITY_MIN_VOTES:
            return dominant
        return center

    # Procesujemy cały obraz, ale wynik zaaplikujemy tylko do maski tła
    filtered_img = generic_filter(
        index_img, 
        _majority_with_threshold, 
        size=cfg.MAJORITY_KERNEL, 
        mode="nearest"
    ).astype(np.uint8)

    # Aplikuj tylko na tle (indeksy 13–127)
    bg_mask = index_img >= 13
    result = index_img.copy()
    result[bg_mask] = filtered_img[bg_mask]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Główna funkcja przetwarzania
# ══════════════════════════════════════════════════════════════════════════════

def process_single(input_path: Path, output_path: Path) -> dict:
    """
    Przetwarza jeden plik mapy i zapisuje TIFF Indexed 127 kolorów.

    Zwraca słownik ze statystykami (utilisable przez 05_process_catalog.py).
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n── {input_path.name} ──")

    # ── Wczytanie palety ──────────────────────────────────────────────────────
    palette_lab, palette_rgb = load_palette()

    ref_lab      = palette_lab[0:12]          # normatywne (12, 3)
    ref_indices  = np.arange(0, 12, dtype=np.uint8)
    cluster_lab  = palette_lab[12:]           # detaliczne (115, 3)
    cluster_idx  = np.arange(12, 127, dtype=np.uint8)
    tree         = cKDTree(cluster_lab)

    # ── Wczytanie obrazu ──────────────────────────────────────────────────────
    img_pil = Image.open(input_path).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.float32) / 255.0   # [H, W, 3] w [0,1]
    h, w, _ = img_rgb.shape
    print(f"  Rozmiar: {w}×{h}  ({w*h:,} pikseli)")

    # ── [0] Opcjonalna korekcja barwna skanera ────────────────────────────────
    if cfg.USE_COLOR_SHIFT:
        print("  [0] Korekcja skanera (color_shift)...")
        img_rgb = apply_color_shift(img_rgb)

    # ── [1] Opcjonalny filtr bilateral ────────────────────────────────────────
    if cfg.USE_BILATERAL:
        print("  [1] Bilateral filter (edge-preserving)...")
        img_rgb = apply_bilateral(img_rgb)

    # ── [2] RGB → LAB ─────────────────────────────────────────────────────────
    print("  [2] RGB → CIELAB...")
    pixels_lab = color.rgb2lab(img_rgb).reshape(-1, 3)

    # ── [3+4] Mapowanie → indeksy ─────────────────────────────────────────────
    print("  [3+4] Mapowanie SNAP + KDTree...")
    indices = map_to_palette(pixels_lab, ref_lab, ref_indices, tree, cluster_idx)

    # ── [5] Opcjonalny majority filter ────────────────────────────────────────
    index_img = indices.reshape(h, w)
    if cfg.USE_MAJORITY_FILTER:
        print("  [5] Majority filter na tle...")
        index_img = apply_majority_filter(index_img)

    # ── [6] Zapis TIFF Indexed ────────────────────────────────────────────────
    print(f"  [6] Zapis → {output_path.name}")
    out_img = Image.fromarray(index_img, mode="P")

    full_palette = np.zeros((256, 3), dtype=np.uint8)
    full_palette[:127] = palette_rgb
    out_img.putpalette(full_palette.flatten().tolist())

    out_img.save(str(output_path), compression="tiff_lzw")

    # ── Statystyki użycia palety ──────────────────────────────────────────────
    # Statystyki użycia palety
    counts = np.bincount(index_img.flatten(), minlength=127)
    used   = int((counts > 0).sum())

    stats = {
        "file": input_path.name,
        "w": w, "h": h,
        "palette_used": used,
        "idx0_pixels": int(counts[0]), # Czarny (normatywny)
    }
    print(f"  Użyto kolorów: {used}/127")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python 04_process_single.py <plik_wejsciowy> [plik_wyjsciowy]")
        sys.exit(1)

    in_path  = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else cfg.OUTPUT_DIR / (in_path.stem + ".tif")
    )

    process_single(in_path, out_path)
    print("\n✓ Etap 04 zakończony. Uruchom 06_validate_single.py aby sprawdzić jakość.")
