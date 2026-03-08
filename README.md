# Pipeline Standaryzacji Kolorów Map Morskich

Ujednolicenie palety barw dla skanów map (TIFF/BMP) przy maksymalnym zachowaniu detali.
Wynik: **TIFF Indexed 127 kolorów**, gotowy pod format nawigacyjny BSB/KAP.

---

## Wymagania

```bash
pip install numpy pillow scikit-image scikit-learn scipy tqdm
```

---

## Struktura katalogów

```
new/
├── maps_raw/          ← 📂 WRZUĆ TU swoje skany (TIFF / BMP)
├── maps_indexed/      ← wynikowe TIFF Indexed (tworzone automatycznie)
├── analysis/          ← próbki LAB i raporty (tworzone automatycznie)
├── palette/           ← paleta kolorów (tworzona automatycznie)
│
├── pipeline_config.py ← ⚙️  JEDYNY plik do edycji
├── 01_analyze_catalog.py
├── 02_build_palette.py
├── 03_validate_palette.py
├── 04_process_single.py
├── 05_process_catalog.py
├── 06_validate_single.py
└── 07_validate_catalog.py
```

---

## Uruchomienie – krok po kroku

> Wszystkie polecenia uruchamiaj z katalogu `new/`.

### Faza 1 – Budowa palety (raz na cały katalog)

```bash
# Krok 1: analiza kolorów wszystkich skanów
python 01_analyze_catalog.py

# Krok 2: budowa globalnej palety 127 kolorów
python 02_build_palette.py

# Krok 3: sprawdzenie poprawności palety
python 03_validate_palette.py
```

Powtarzaj fazę 1 tylko gdy **dodasz nowe skany** do katalogu lub zmienisz kolory normatywne.

---

### Faza 2 – Konwersja map

```bash
# Opcja A: test jednej mapy (polecane przed batch)
python 04_process_single.py maps_raw/nazwa_pliku.tif

# Opcja B: przetworzenie całego katalogu
python 05_process_catalog.py
```

Wyniki trafiają do `maps_indexed/`.

---

### Faza 3 – Walidacja jakości

```bash
# Sprawdzenie jednej mapy (ΔE + SSIM + krawędzie)
python 06_validate_single.py maps_raw/plik.tif maps_indexed/plik.tif

# Walidacja batch → raport CSV
python 07_validate_catalog.py
```

---

## Konfiguracja (`pipeline_config.py`)

Edytuj **tylko ten plik** – nie zmieniaj nic wewnątrz pozostałych skryptów.

### Najważniejsze parametry

| Parametr | Domyślnie | Co robi |
|---|---|---|
| Parametr | Domyślnie | Co robi |
|---|---|---|
| `USE_BILATERAL` | `True` | Filtr wygładzający powierzchnie (np. ziarno skanera) przy zachowaniu ostrych krawędzi detali. |
| `BILATERAL_SIGMA_COLOR` | `0.05` | Czułość na różnicę kolorów – większa wartość mocniej ujednolica podobne barwy. |
| `BILATERAL_SIGMA_SPATIAL` | `3` | Zasięg przestrzenny wygładzania w pikselach. |
| `SNAP_THRESHOLD` | `6.0` | Tolerancja (ΔE) – piksele barwnie bliskie kolorom normatywnym zostaną do nich dociągnięte. |
| `USE_MAJORITY_FILTER` | `True` | Filtr "większościowy" – usuwa drobny szum (pojedyncze piksele) w obszarach tła. |
| `USE_COLOR_SHIFT` | `False` | Korekcja balansu bieli i barw skanera (wymaga profilu w `color_defs.py`). |
| `USE_SEGMENTED_PALETTE` | `False` | `False` optymalizuje całą paletę naraz; `True` wymusza podział klastrów wg jasności. |

### Kolory normatywne

Lista `NORMATIVE_COLORS_RGB` definiuje 12 kluczowych kolorów (woda, ląd, światła, izobaty), które otrzymują **rezerwowane indeksy 0–11**. Są one chronione przed przypadkowym nadpisaniem przez inne odcienie podczas budowy palety.

System walidacji priorytetyzuje **bezpieczeństwo nawigacyjne** (strukturę) nad idealną wiernością barw tła.

| Kategoria | Metryki | Próg | Waga |
|---|---|---|---|
| **STRUKTURA** | SSIM, Edge IoU, Edge Dist | >=0.9 / >=0.8 / <=1.0px | **CRITICAL** (FAIL) |
| **NORMATYWNE** | Max ΔE dla świateł/izobat | <= 1.0 | **CRITICAL** (FAIL) |
| **KOLOR TŁA** | Mean ΔE, % ΔE > 10 | <= 2.0 / <= 0.2% | **WARNING** |

Status **WARNING** oznacza, że mapa jest w pełni czytelna i bezpieczna, ale jej ogólne zabarwienie odbiega od oryginału.

---

## Korekcja barwna skanera (opcjonalna)

Jeśli skaner systematycznie przesuwa kolory (np. wszystko za ciepłe), możesz użyć korekcji:

1. Stwórz plik `color_defs.py` wzorując się na `../color_defs.py`
2. Skalibruj pary `scan_colors` ↔ `colors` dla swojego skanera
3. Ustaw `USE_COLOR_SHIFT = True` w `pipeline_config.py`

Korekcja działa w przestrzeni **RGB (addytywnie)** i jest bezpieczna dla detali. Użyj jej, jeśli Twoje skany mają stałe zabarwienie (np. są zbyt "ciepłe").

---

| Objaw | Przyczyna | Rozwiązanie |
|---|---|---|
| Błąd `Brak palety` | Pominięto kroki 1–3 | Uruchom `01 → 02 → 03` |
| STATUS: FAIL | Rozmazane napisy / błędne światła | Zmniejsz `BILATERAL_SIGMA_SPATIAL` lub popraw paletę |
| STATUS: WARNING | Kolor tła "pływa" | Dopuszczalne, o ile struktura jest PASS |
| ΔE > 2 w wodzie | Słabe próbkowanie jasnych tonów | Zwiększ `BRIGHT_SAMPLING_RATIO` w konfigu |
