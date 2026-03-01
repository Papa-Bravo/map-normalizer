# System Standaryzacji Kolorów dla Map Morskich

## TIFF → 127 kolorów → KAP-ready

Ten dokument opisuje kompletny pipeline przetwarzania kolorystycznego
map morskich, od analizy barw źródłowych do wygenerowania
zestandaryzowanych TIFF-ów gotowych pod BSB/KAP.

------------------------------------------------------------------------

# Wymagania

Python 3.8+

Biblioteki:

    pip install numpy pillow scikit-image scikit-learn scipy tqdm

------------------------------------------------------------------------

# Struktura katalogów

maps_raw/ → źródłowe TIFF 24-bit\
analysis/ → próbki LAB i raport analizy\
palette/ → wygenerowana paleta 127 kolorów\
maps_indexed_v3/ → wynikowe TIFF indexed

------------------------------------------------------------------------

# ETAP 1 --- Analiza kolorów źródłowych

## Skrypt: analyze_dataset.py

Uruchomienie:

    python analyze_dataset.py

Działanie:

1.  Wczytuje wszystkie TIFF z katalogu `maps_raw/`

2.  Usuwa czystą biel (ramki skanów)

3.  Losowo próbuje 3% pikseli

4.  Ogranicza nadreprezentację bardzo jasnych tonów (L\* \> 95)

5.  Konwertuje RGB → CIELAB

6.  Zapisuje:

    analysis/sample_lab.npy analysis/report.txt

Cel:

Uzyskanie reprezentatywnego zbioru kolorów do budowy palety.

------------------------------------------------------------------------

# ETAP 2 --- Budowa palety 127 kolorów (KAP-ready)

## Skrypt: build_palette_v3_127.py

Uruchomienie:

    python build_palette_v3_127.py

Działanie:

1.  Wczytuje `analysis/sample_lab.npy`

2.  Usuwa piksele bliskie 12 kolorom referencyjnym

3.  Dzieli przestrzeń na segmenty L\*

4.  Wykonuje MiniBatchKMeans per segment

5.  Składa paletę:

    indeks 0 → RESERVED indeksy 1--12 → kolory referencyjne indeksy
    13--127 → centroidy klastrów

Zapisuje:

    palette/palette_v3_127_lab.npy
    palette/palette_v3_127_rgb.npy
    palette/palette_v3_127_rgb.csv
    palette/palette_v3_127_preview.png

------------------------------------------------------------------------

# ETAP 3 --- Walidacja palety

## 3.1 Walidacja kolorów referencyjnych

Skrypt: palette_validator_v3.py

    python palette_validator_v3.py

Sprawdza ΔE dla indeksów 1--12. Oczekiwane: ΔE \< 1.

------------------------------------------------------------------------

## 3.2 Test pojedynczej mapy

Skrypt: apply_palette_v3_single.py

    python apply_palette_v3_single.py

Działanie:

1.  RGB → LAB
2.  SNAP do referencyjnych (ΔE \< 5)
3.  KDTree dla 13--127
4.  Zapis TIFF Indexed + LZW
5.  Raport jakości ΔE

Kryteria:

-   Średni ΔE \< 2
-   \% ΔE \> 10 minimalny

------------------------------------------------------------------------

# ETAP 4 --- Batch konwersja wszystkich map

Skrypt: apply_palette_v3_batch.py

    python apply_palette_v3_batch.py

Działanie:

-   Przetwarza cały katalog `maps_raw/`

-   Zapisuje wynik do `maps_indexed_v3/`

-   Generuje:

    maps_indexed_v3/batch_report.csv

Interpretacja:

-   Średni ΔE \< 2 → bardzo dobry
-   \% ΔE \> 10 \< 0.2% → bezpieczny

------------------------------------------------------------------------

# ETAP 5 --- Weryfikacja użycia indeksów

## Pojedyncza mapa

Skrypt: verify_palette_usage_v3.py

    python verify_palette_usage_v3.py

Sprawdza:

-   Czy indeks 0 jest nieużywany
-   Ile kolorów 1--127 jest wykorzystywanych

------------------------------------------------------------------------

## Batch

Skrypt: verify_palette_usage_v3_batch.py

    python verify_palette_usage_v3_batch.py

Generuje:

    maps_indexed_v3/palette_usage_report.csv

Kryteria poprawności:

-   Wykorzystanie palety \> 90%
-   Indeks 0 = 0 pikseli

------------------------------------------------------------------------

# Ograniczenia BSB/KAP

-   7 bitów na piksel
-   Maksymalnie 127 kolorów
-   Indeks 0 zarezerwowany

Paleta v3_127 spełnia te wymagania.

------------------------------------------------------------------------

# Kompletny Pipeline

1.  python analyze_dataset.py
2.  python build_palette_v3_127.py
3.  python palette_validator_v3.py
4.  python apply_palette_v3_single.py
5.  python apply_palette_v3_batch.py
6.  python verify_palette_usage_v3_batch.py

------------------------------------------------------------------------

# Status systemu

Paleta v3_127:

-   matematycznie zwalidowana
-   zgodna z wymaganiami BSB/KAP
-   efektywnie wykorzystywana (\~95%)
-   gotowa do produkcyjnego generowania KAP
