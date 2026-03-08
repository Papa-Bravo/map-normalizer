# COLOR PIPELINE NOTES

## 1. Context

**Project:** Standaryzacja kolorów map morskich (TIFF → wspólna paleta →
TIFF Indexed)\
**Cel:** Deterministyczna spójność kolorystyczna między atlasami przy
zachowaniu wysokiej wierności wizualnej.

------------------------------------------------------------------------

## 2. Dataset Summary (ETAP 1)

-   Liczba map: 14\
-   Łączna liczba pikseli: 486,973,720\
-   Odrzucone czyste biele: 283,329,509 (\~58%)\
-   Próbka LAB: 3,414,701 pikseli (\~82 MB)\
-   Rozkład L\*: mean ≈ 83 (mapy ogólnie jasne)\
-   Delikatne ciepłe tony (b\* \> 0)

**Wniosek:** Próbka reprezentatywna, brak potrzeby dodatkowej filtracji.

------------------------------------------------------------------------

## 3. Assumptions

1.  Czysta biel (RGB \> 250) to artefakt skanu / ramka.
2.  Jasne obszary morza są istotne i muszą pozostać w palecie.
3.  256 kolorów powinno być wystarczające dla map morskich.
4.  Konwersja do LAB daje lepszą percepcyjną klasteryzację niż RGB.
5.  Docelowa paleta musi być:
    -   globalna,
    -   wersjonowana,
    -   deterministyczna.

------------------------------------------------------------------------

## 4. Decisions

### 4.1 Klasteryzacja

-   Algorytm: MiniBatchKMeans\
-   k = 256\
-   random_state = 42\
-   batch_size = 10_000\
-   max_iter = 200

**Powód:**\
3.4 mln próbek → klasyczny KMeans byłby kosztowny.\
MiniBatch zapewnia stabilny wynik przy dużej liczbie punktów oraz
deterministykę przez seed.

------------------------------------------------------------------------

### 4.2 Przestrzeń koloru

-   Klasteryzacja w LAB\
-   Paleta zapisana jako RGB\
-   Mapowanie do palety w LAB (minimalizacja ΔE)

------------------------------------------------------------------------

### 4.3 Dithering

Na etapie budowy palety: brak ditheringu.\
Decyzja o ewentualnym ditheringu po testach jakości.

------------------------------------------------------------------------

## 5. Open Questions / TODO

-   [ ] Czy 256 kolorów wystarczy? (analiza ΔE max / mean)
-   [ ] Czy jasne odcienie morza zajmują zbyt wiele klastrów?
-   [ ] Czy potrzebna będzie korekta palety (manualne łączenie podobnych
    klastrów)?
-   [ ] Czy generowanie KAP wymaga dodatkowych ograniczeń palety?

------------------------------------------------------------------------

## 6. Risks

1.  Dominacja bardzo jasnych tonów.
2.  Zbyt mała rozdzielczość palety dla cienkich linii.
3.  Nowe mapy w przyszłości mogą wymagać przebudowy palety.

------------------------------------------------------------------------

## 7. Versioning Strategy

Paleta będzie wersjonowana:

-   palette_v1\
-   palette_v2 (jeśli zmienimy k lub algorytm)

Każdy TIFF indexed powinien zawierać informację o wersji palety użytej
do konwersji.

------------------------------------------------------------------------

*Generated on: 2026-03-01T09:58:58.244555*
