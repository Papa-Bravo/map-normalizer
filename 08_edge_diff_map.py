"""
ETAP 08 – Generowanie mapy różnic krawędzi (Edge Difference Map).

Wizualizuje zmiany strukturalne (linie, napisy, kontury) między oryginałem a wynikiem.
Używa detekcji krawędzi Canny.

Użycie:
    python 08_edge_diff_map.py <oryginalny> <przetworzony> [plik_wyjsciowy]

Legenda:
    BIAŁY  = Krawędzie wspólne (zachowane)
    CZERWONY = Krawędzie utracone (obecne tylko w oryginale)
    NIEBIESKI = Krawędzie nowe (wprowadzone przez procesowanie)
    CZARNY = Brak krawędzi
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import color
from skimage.feature import canny

# Wpięcie ścieżki dla importu configu
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg

def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] → szarość float [0,1]."""
    return 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]

def generate_edge_diff(orig_path: Path, proc_path: Path, out_path: Path):
    # 1. Load images
    orig_pil = Image.open(orig_path).convert("RGB")
    proc_pil = Image.open(proc_path).convert("RGB")
    
    orig_rgb = np.array(orig_pil, dtype=np.float32) / 255.0
    proc_rgb = np.array(proc_pil, dtype=np.float32) / 255.0
    
    # 2. Verify dimensions
    if orig_rgb.shape != proc_rgb.shape:
        print(f"[BŁĄD] Różne rozmiary: {orig_rgb.shape[:2]} vs {proc_rgb.shape[:2]}")
        sys.exit(1)
    
    # 3. Grayscale
    g_orig = _to_gray(orig_rgb)
    g_proc = _to_gray(proc_rgb)
    
    # 4. Canny Edge Detection
    # Przeliczenie progów 0-255 na 0-1 (skimage canny oczekuje float w [0,1] dla float input)
    low = cfg.EDGE_CANNY_LOW / 255.0
    high = cfg.EDGE_CANNY_HIGH / 255.0
    
    print(f"Detekcja krawędzi Canny (low={cfg.EDGE_CANNY_LOW}, high={cfg.EDGE_CANNY_HIGH})...")
    e_orig = canny(g_orig, low_threshold=low, high_threshold=high)
    e_proc = canny(g_proc, low_threshold=low, high_threshold=high)
    
    # 5. Compute binary maps
    lost   = e_orig & ~e_proc
    new    = e_proc & ~e_orig
    common = e_orig & e_proc
    
    # 6. Visualization
    h, w = g_orig.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # White = common
    vis[common] = [255, 255, 255]
    # Red = lost
    vis[lost] = [255, 0, 0]
    # Blue = new
    vis[new] = [0, 0, 255]
    
    # Save
    Image.fromarray(vis).save(out_path)
    print(f"Zapisano wizualizację: {out_path}")
    
    # 7. Statistics
    n_orig = int(np.sum(e_orig))
    n_proc = int(np.sum(e_proc))
    n_common = int(np.sum(common))
    n_lost = int(np.sum(lost))
    n_new = int(np.sum(new))
    
    print("\n## Edge statistics")
    print(f"Original edges:  {n_orig}")
    print(f"Processed edges: {n_proc}")
    print(f"Preserved edges: {n_common}")
    print(f"Lost edges:      {n_lost}")
    print(f"New edges:       {n_new}")
    
    if n_orig > 0:
        print(f"\nLost edge ratio: {100.0 * n_lost / n_orig:.2f}%")
        print(f"New edge ratio:  {100.0 * n_new / n_orig:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Użycie: python 08_edge_diff_map.py <oryginalny> <przetworzony> [wynik.png]")
        sys.exit(1)
    
    p_orig = Path(sys.argv[1])
    p_proc = Path(sys.argv[2])
    p_out  = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("edge_diff_map.png")
    
    generate_edge_diff(p_orig, p_proc, p_out)
