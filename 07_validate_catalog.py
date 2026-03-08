"""
ETAP 07 – Batch walidacja całego katalogu przetworzonych map.

URUCHAMIAJ NA CAŁYM KATALOGU (po etapie 05).

Porównuje każdy plik z OUTPUT_DIR z jego oryginałem z INPUT_DIR.
Generuje raport CSV: OUTPUT_DIR/validation_report.csv

Kryteria PASS (konfigurowalne w pipeline_config.py):
  Kolorystyczne : mean_de ≤ 2.0,  pct_gt10 ≤ 0.2%
  Strukturalne  : SSIM ≥ 0.90,  edge_iou ≥ 0.80,
                  grad_corr ≥ 0.90,  edge_dist ≤ 1.0 px
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import importlib.util
from pathlib import Path
from tqdm import tqdm
import pipeline_config as cfg

# Dynamiczny import 06_validate_single.py
_spec = importlib.util.spec_from_file_location(
    "validate_single_mod",
    Path(__file__).parent / "06_validate_single.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
validate_single = _mod.validate_single

# ── Parowanie oryginał ↔ przetworzony ────────────────────────────────────────

pairs = []
for proc_path in sorted(cfg.OUTPUT_DIR.glob("*.tif")):
    # Szukaj oryginału – dowolny wspierany format z tą samą nazwą bazową
    orig = None
    for ext in cfg.SUPPORTED_EXTENSIONS:
        candidate = cfg.INPUT_DIR / (proc_path.stem + ext)
        if candidate.exists():
            orig = candidate
            break
    if orig:
        pairs.append((orig, proc_path))
    else:
        print(f"[UWAGA] Brak oryginału dla {proc_path.name} – pomijam.")

if not pairs:
    print(f"[BŁĄD] Brak par do walidacji. Uruchom najpierw 05_process_catalog.py")
    sys.exit(1)

print(f"\nWalidacja: {len(pairs)} par\n")

# ── Walidacja ─────────────────────────────────────────────────────────────────

results = []

for orig, proc in tqdm(pairs, desc="Walidacja"):
    try:
        r = validate_single(orig, proc)
        results.append(r)
    except Exception as exc:
        print(f"\n[BŁĄD] {orig.name}: {exc}")
        results.append({
            "file": orig.name,
            "mean_de": None, "max_de": None,
            "pct_gt5": None, "pct_gt10": None,
            "color_ok": False,
            "ssim": None, "edge_iou": None,
            "grad_corr": None, "edge_dist": None, "edge_within2": None,
            "struct_ok": False,
            "norm_ok": False, "max_norm_de": None,
            "status": "ERROR",
            "pass": False,
        })

# ── Raport CSV ────────────────────────────────────────────────────────────────

report_path = cfg.OUTPUT_DIR / "validation_report.csv"
fieldnames = [
    "file", "status",
    "mean_de", "max_de", "pct_gt10", "color_ok",
    "ssim", "edge_iou", "grad_corr", "edge_dist", "struct_ok",
    "norm_ok", "max_norm_de", "pass"
]

with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                         for k in fieldnames})

# ── Podsumowanie ──────────────────────────────────────────────────────────────

n_pass    = sum(1 for r in results if r["status"] == "PASS")
n_warning = sum(1 for r in results if r["status"] == "WARNING")
n_fail    = sum(1 for r in results if r["status"] == "FAIL")
n_error   = sum(1 for r in results if r["status"] == "ERROR")
n         = len(results)

print(f"\n===== RAPORT WALIDACJI BATCH =====")
print(f"  Łącznie plików     : {n}")
print(f"  STATUS: PASS       : {n_pass}")
print(f"  STATUS: WARNING    : {n_warning} (tylko błędy koloru)")
print(f"  STATUS: FAIL       : {n_fail} (błędy struktury/normatywne)")
if n_error > 0:
    print(f"  STATUS: ERROR      : {n_error} (błędy wykonania)")
print(f"  Raport CSV         : {report_path}")

attention = [r for r in results if r["status"] != "PASS"]
if attention:
    print(f"\nPliki wymagające uwagi ({len(attention)}):")
    for r in attention:
        issues = []
        if r["status"] == "ERROR":
            issues.append("Błąd przetwarzania")
        else:
            if not r["struct_ok"]: issues.append("STRUKTURA")
            if not r["norm_ok"]:   issues.append(f"NORMATYWNE(ΔE={r['max_norm_de']:.2f})")
            if not r["color_ok"]:  issues.append(f"KOLOR(ΔE={r['mean_de']:.2f})")
        
        label = f"[{r['status']}]"
        print(f"  {label:10s} {r['file']:40s}  problemy: {', '.join(issues)}")
else:
    print("\n✓ Wszystkie pliki spełniają kryteria jakości (PASS).")
