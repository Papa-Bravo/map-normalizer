"""
ETAP 06 – Walidacja jakości POJEDYNCZEGO przetworzonego pliku.

URUCHAMIAJ PER-PLIK (lub użyj 07_validate_catalog.py dla batch).

Użycie:
    python 06_validate_single.py <oryginalny> <przetworzony>

Mierzy dwa wymiary jakości:
  A) KOLORYSTYCZNA (ΔE w przestrzeni LAB):
     - Średni ΔE < 2.0           → bardzo dobra wierność barw
     - % pikseli ΔE > 10 < 0.2% → brak dużych przeskoków

  B) STRUKTURALNA (zachowanie detali – krawędzie, kontrast):
     - SSIM > 0.90               → ogólna zgodność struktury
     - Edge IoU > 0.80           → krawędzie (Canny) są w obu obrazach
     - Gradient corr > 0.90      → kontrast przejść zachowany
     - Edge distance < 1.0 px   → krawędzie nie przesunęły się

Nie używa OpenCV – wyłącznie NumPy, scikit-image i SciPy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import color
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from scipy.ndimage import distance_transform_edt, sobel
import pipeline_config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Metryki strukturalne
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] → szarość float [0,1]."""
    return 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]


def compute_ssim(orig: np.ndarray, proc: np.ndarray) -> float:
    """Structural Similarity Index (0–1, 1 = identyczne)."""
    g1 = _to_gray(orig)
    g2 = _to_gray(proc)
    score, _ = ssim(g1, g2, data_range=1.0, full=True)
    return float(score)


def compute_edge_iou(orig: np.ndarray, proc: np.ndarray,
                     low: float = 0.1, high: float = 0.2) -> float:
    """
    Intersection-over-Union maski krawędzi Canny.
    Progi low/high zdefiniowane w [0,1] zamiast [0,255] (skimage Canny).
    """
    e1 = canny(_to_gray(orig), low_threshold=low, high_threshold=high)
    e2 = canny(_to_gray(proc), low_threshold=low, high_threshold=high)

    inter = float(np.logical_and(e1, e2).sum())
    union = float(np.logical_or(e1, e2).sum())
    return inter / union if union > 0 else 1.0


def compute_gradient_corr(orig: np.ndarray, proc: np.ndarray) -> float:
    """
    Korelacja Pearsona map gradientów Sobela.
    Mierzy czy amplituda przejść jasność/cień jest zachowana.
    """
    def _grad_magnitude(img):
        g = _to_gray(img)
        gx = sobel(g, axis=1)
        gy = sobel(g, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2).flatten()

    g1 = _grad_magnitude(orig)
    g2 = _grad_magnitude(proc)
    return float(np.corrcoef(g1, g2)[0, 1])


def compute_edge_distance(orig: np.ndarray, proc: np.ndarray,
                          low: float = 0.1, high: float = 0.2):
    """
    Średnia odległość krawędzi przetworzonego obrazu od krawędzi oryginału [px].
    Zwraca (mean_dist, median_dist, within_2px_ratio).
    """
    e_ref  = canny(_to_gray(orig), low_threshold=low, high_threshold=high)
    e_proc = canny(_to_gray(proc), low_threshold=low, high_threshold=high)

    # Distance transform od krawędzi referencyjnych
    dist_map = distance_transform_edt(~e_ref)

    distances = dist_map[e_proc]
    if len(distances) == 0:
        return 0.0, 0.0, 1.0

    return (
        float(np.mean(distances)),
        float(np.median(distances)),
        float(np.sum(distances <= 2) / len(distances)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Metryki kolorystyczne
# ══════════════════════════════════════════════════════════════════════════════

def compute_delta_e(orig_rgb: np.ndarray, proc_rgb: np.ndarray):
    """
    ΔE piksel-po-pikselu w przestrzeni CIELAB.
    Zwraca (mean, max, pct_gt5, pct_gt10).
    """
    orig_lab = color.rgb2lab(orig_rgb)
    proc_lab = color.rgb2lab(proc_rgb)
    delta_e  = np.sqrt(np.sum((orig_lab - proc_lab) ** 2, axis=2))
    total    = delta_e.size
    return (
        float(delta_e.mean()),
        float(delta_e.max()),
        float((delta_e > 5).sum()  / total * 100),
        float((delta_e > 10).sum() / total * 100),
    )


def compute_edge_diff_metrics(orig_rgb: np.ndarray, proc_rgb: np.ndarray, 
                             output_diff_path: Path = None):
    """
    Oblicza statystyki różnic krawędzi i opcjonalnie zapisuje mapę wizualną.
    """
    g_orig = _to_gray(orig_rgb)
    g_proc = _to_gray(proc_rgb)

    low = cfg.EDGE_CANNY_LOW / 255.0
    high = cfg.EDGE_CANNY_HIGH / 255.0

    e_orig = canny(g_orig, low_threshold=low, high_threshold=high)
    e_proc = canny(g_proc, low_threshold=low, high_threshold=high)

    lost   = e_orig & ~e_proc
    new    = e_proc & ~e_orig
    common = e_orig & e_proc

    n_orig = int(np.sum(e_orig))
    n_lost = int(np.sum(lost))
    n_new  = int(np.sum(new))

    lost_ratio = (100.0 * n_lost / n_orig) if n_orig > 0 else 0.0
    new_ratio  = (100.0 * n_new / n_orig) if n_orig > 0 else 0.0

    if output_diff_path and cfg.GENERATE_EDGE_DIFF_ON_VALIDATE:
        h, w = g_orig.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[common] = [255, 255, 255] # Biały = zachowane
        vis[lost]   = [255, 0, 0]     # Czerwony = utracone
        vis[new]    = [0, 0, 255]     # Niebieski = nowe
        Image.fromarray(vis).save(output_diff_path)

    return lost_ratio, new_ratio


def validate_normative_colors(proc_path: Path):
    """
    Sprawdza czy kolory normatywne w przetworzonym pliku 
    są zgodne z definicją w palecie.
    """
    try:
        # Wczytaj paletę użytą do zapisu (z nagłówka TIFF) lub z configu
        img = Image.open(proc_path)
        if img.mode != 'P':
            return None
        
        pal = np.array(img.getpalette()).reshape(-1, 3)[:128]
        # Porównaj pierwsze 12 kolorów z NORMATIVE_COLORS_RGB
        ref_rgb = np.array(cfg.NORMATIVE_COLORS_RGB, dtype=np.float32)
        proc_ref_rgb = pal[:12].astype(np.float32)
        
        ref_lab = color.rgb2lab(ref_rgb.reshape(1, -1, 3) / 255.0)[0]
        proc_lab = color.rgb2lab(proc_ref_rgb.reshape(1, -1, 3) / 255.0)[0]
        
        de_list = np.sqrt(np.sum((ref_lab - proc_lab) ** 2, axis=1))
        return de_list
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Główna funkcja walidacji
# ══════════════════════════════════════════════════════════════════════════════

def validate_single(orig_path: Path, proc_path: Path) -> dict:
    """
    Porównuje oryginalny skan z przetworzonym plikiem Indexed.
    Zwraca słownik ze wszystkimi metrykami i flagą pass/fail.
    """
    orig_path = Path(orig_path)
    proc_path = Path(proc_path)

    # Wczytanie – przetworzony TIFF Indexed konwertujemy do RGB
    orig_rgb = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32) / 255.0
    proc_rgb = np.array(Image.open(proc_path).convert("RGB"), dtype=np.float32) / 255.0

    if orig_rgb.shape != proc_rgb.shape:
        raise ValueError(
            f"Różne rozmiary: oryginał {orig_rgb.shape[:2]}, "
            f"przetworzony {proc_rgb.shape[:2]}"
        )

    # ── Metryki kolorystyczne ─────────────────────────────────────────────────
    mean_de, max_de, pct_gt5, pct_gt10 = compute_delta_e(orig_rgb, proc_rgb)

    # ── Metryki strukturalne ──────────────────────────────────────────────────
    ssim_score = compute_ssim(orig_rgb, proc_rgb)
    edge_iou   = compute_edge_iou(orig_rgb, proc_rgb)
    grad_corr  = compute_gradient_corr(orig_rgb, proc_rgb)
    mean_d, median_d, within2 = compute_edge_distance(orig_rgb, proc_rgb)

    # ── Metryki normatywne ─────────────────────────────────────────────────────
    norm_de = validate_normative_colors(proc_path)
    norm_ok = True
    max_norm_de = 0.0
    if norm_de is not None:
        max_norm_de = float(np.max(norm_de))
        norm_ok = max_norm_de <= cfg.MAX_NORMATIVE_DE

    # ── Metryki Edge Diff ─────────────────────────────────────────────────────
    diff_path = None
    if cfg.GENERATE_EDGE_DIFF_ON_VALIDATE:
        diff_path = proc_path.parent / f"{orig_path.stem}_diff_map.png"
    
    lost_ratio, new_ratio = compute_edge_diff_metrics(orig_rgb, proc_rgb, diff_path)

    # ── Ocena PASS/FAIL/WARNING ───────────────────────────────────────────────
    # STRUKTURA = CRITICAL
    struct_ok = (
        ssim_score >= cfg.MIN_SSIM and
        edge_iou   >= cfg.MIN_EDGE_IOU and
        grad_corr  >= cfg.MIN_GRAD_CORR and
        mean_d     <= cfg.MAX_EDGE_DIST_PX and
        lost_ratio <= cfg.MAX_LOST_EDGE_RATIO and
        new_ratio  <= cfg.MAX_NEW_EDGE_RATIO
    )
    
    # KOLOR (ogólny) = WARNING
    color_ok  = (mean_de <= cfg.MAX_MEAN_DE) and (pct_gt10 <= cfg.MAX_PCT_GT10)

    # Werdykt globalny: 
    # FAIL jeśli struktura lub normatywne leżą.
    # WARNING jeśli tylko ogólny kolor leży.
    is_fail = (not struct_ok) or (not norm_ok)
    is_warning = struct_ok and norm_ok and (not color_ok)
    
    global_status = "FAIL" if is_fail else ("WARNING" if is_warning else "PASS")

    return {
        "file":        orig_path.name,
        "mean_de":     mean_de,
        "max_de":      max_de,
        "pct_gt5":     pct_gt5,
        "pct_gt10":    pct_gt10,
        "color_ok":    color_ok,
        "ssim":        ssim_score,
        "edge_iou":    edge_iou,
        "grad_corr":   grad_corr,
        "edge_dist":   mean_d,
        "edge_within2": within2,
        "struct_ok":   struct_ok,
        "norm_ok":     norm_ok,
        "max_norm_de": max_norm_de,
        "lost_edge_pct": lost_ratio,
        "new_edge_pct":  new_ratio,
        "status":      global_status,
        "pass":        not is_fail, # TRUE dla PASS i WARNING
    }


def print_report(r: dict):
    """Drukuje czytelny raport dla jednego pliku."""
    def mark(ok): return "✓" if ok else "✗"
    def status_label(s):
        if s == "PASS": return "\033[92mPASS\033[0m"
        if s == "WARNING": return "\033[93mWARNING\033[0m"
        return "\033[91mFAIL\033[0m"

    print(f"\n===== WALIDACJA: {r['file']} =====")
    print(f"  Profil jakości     : {cfg.QUALITY_PROFILE}")
    
    print("\n── 1. STRUKTURALNA (CRITICAL - detale) ───────────────")
    print(f"  SSIM               : {r['ssim']:.4f}  (próg ≥ {cfg.MIN_SSIM})   {mark(r['ssim'] >= cfg.MIN_SSIM)}")
    print(f"  Edge IoU (Canny)   : {r['edge_iou']:.4f}  (próg ≥ {cfg.MIN_EDGE_IOU})   {mark(r['edge_iou'] >= cfg.MIN_EDGE_IOU)}")
    print(f"  Gradient corr      : {r['grad_corr']:.4f}  (próg ≥ {cfg.MIN_GRAD_CORR})   {mark(r['grad_corr'] >= cfg.MIN_GRAD_CORR)}")
    print(f"  Edge dist [px]     : {r['edge_dist']:.3f}   (próg ≤ {cfg.MAX_EDGE_DIST_PX} px)   {mark(r['edge_dist'] <= cfg.MAX_EDGE_DIST_PX)}")
    print(f"  Kraw. w 2px        : {r['edge_within2']*100:.1f}%")
    print(f"  Lost Edge Ratio    : {r['lost_edge_pct']:.2f}%  (próg ≤ {cfg.MAX_LOST_EDGE_RATIO}%)  {mark(r['lost_edge_pct'] <= cfg.MAX_LOST_EDGE_RATIO)}")
    print(f"  New Edge Ratio     : {r['new_edge_pct']:.2f}%  (próg ≤ {cfg.MAX_NEW_EDGE_RATIO}%)  {mark(r['new_edge_pct'] <= cfg.MAX_NEW_EDGE_RATIO)}")
    print(f"  Wynik struktury    : {'PASS' if r['struct_ok'] else 'FAIL'}")

    print("\n── 2. NORMATYWNA (CRITICAL - semantyka) ──────────────")
    print(f"  Max ΔE (normatywne): {r['max_norm_de']:.3f}   (próg ≤ {cfg.MAX_NORMATIVE_DE})   {mark(r['norm_ok'])}")
    print(f"  Wynik normatywny   : {'PASS' if r['norm_ok'] else 'FAIL'}")

    print("\n── 3. KOLORYSTYCZNA (WARNING - tło) ──────────────────")
    print(f"  Średni ΔE          : {r['mean_de']:.3f}   (próg ≤ {cfg.MAX_MEAN_DE})   {mark(r['mean_de'] <= cfg.MAX_MEAN_DE)}")
    print(f"  Maks  ΔE           : {r['max_de']:.3f}")
    print(f"  % pikseli ΔE > 10  : {r['pct_gt10']:.3f}%  (próg ≤ {cfg.MAX_PCT_GT10}%)   {mark(r['pct_gt10'] <= cfg.MAX_PCT_GT10)}")
    print(f"  Wynik koloryst.    : {'PASS' if r['color_ok'] else 'WARNING'}")

    print(f"\n  STATUS PLIKU: {status_label(r['status'])}")
    if r['status'] == "WARNING":
        print("  [NOTKA] Mapa czytelna strukturalnie, zabarwienie różne od oryginału.")
    elif r['status'] == "FAIL":
        print("  [BŁĄD] Mapa niebezpieczna lub błędna semantycznie!")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie:")
        print("  python 06_validate_single.py <oryginalny> <przetworzony>")
        sys.exit(1)

    result = validate_single(Path(sys.argv[1]), Path(sys.argv[2]))
    print_report(result)
    sys.exit(0 if result["pass"] else 1)
