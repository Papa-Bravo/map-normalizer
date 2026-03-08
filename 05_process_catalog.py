"""
ETAP 05 – Batch przetwarzanie całego katalogu.

URUCHAMIAJ NA CAŁYM KATALOGU (po etapach 01–03).

Przetwarza wszystkie pliki TIFF/BMP z INPUT_DIR przy użyciu logiki z 04_process_single.py.
Wyniki zapisuje do OUTPUT_DIR.
Generuje raport CSV: OUTPUT_DIR/batch_report.csv

Kryteria jakości w raporcie:
  palette_used > 50%     → typowe wykorzystanie barw dla map
  palette_pct            → procentowe wykorzystanie dostępnych 127 kolorów
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import importlib.util
from pathlib import Path
from tqdm import tqdm
import pipeline_config as cfg

# Dynamiczny import 04_process_single.py (cyfra na początku = niestandardowa nazwa)
_spec = importlib.util.spec_from_file_location(
    "process_single_mod",
    Path(__file__).parent / "04_process_single.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
process_single = _mod.process_single

# ── Zbieranie plików ──────────────────────────────────────────────────────────

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_files = [
    p for p in sorted(cfg.INPUT_DIR.iterdir())
    if p.suffix.lower() in cfg.SUPPORTED_EXTENSIONS
]

if not input_files:
    print(f"[BŁĄD] Brak plików w {cfg.INPUT_DIR}")
    sys.exit(1)

print(f"\nRozpoczynam batch: {len(input_files)} plików → {cfg.OUTPUT_DIR}\n")

# ── Przetwarzanie ─────────────────────────────────────────────────────────────

results = []
errors  = []

for path in tqdm(input_files, desc="Przetwarzanie"):
    out_path = cfg.OUTPUT_DIR / (path.stem + ".tif")
    try:
        stats = process_single(path, out_path)
        results.append(stats)
    except Exception as exc:
        print(f"\n[BŁĄD] {path.name}: {exc}")
        errors.append({"file": path.name, "error": str(exc)})

# ── Raport CSV ────────────────────────────────────────────────────────────────

report_path = cfg.OUTPUT_DIR / "batch_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "w", "h", "palette_used", "idx0_pixels", "palette_pct"]
    )
    writer.writeheader()
    for r in results:
        r["palette_pct"] = f"{100 * r['palette_used'] / 127:.1f}"
        writer.writerow(r)

# ── Podsumowanie ──────────────────────────────────────────────────────────────

ok_palette = sum(1 for r in results if r["palette_used"] / 127 >= 0.50)

print(f"\n===== RAPORT BATCH =====")
print(f"  Przetworzono  : {len(results)} / {len(input_files)}")
print(f"  Błędów        : {len(errors)}")
print(f"  Paleta > 50%  : {ok_palette} / {len(results)}")
print(f"  Raport CSV    : {report_path}")

if errors:
    print("\nPliki z błędami:")
    for e in errors:
        print(f"  {e['file']}: {e['error']}")

print("\n✓ Etap 05 zakończony. Uruchom 07_validate_catalog.py aby sprawdzić jakość.")
