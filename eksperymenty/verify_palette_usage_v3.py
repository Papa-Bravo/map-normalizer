import numpy as np
from PIL import Image

INDEXED_IMAGE_PATH = "maps_indexed_v3/1020_00.tif"

img = Image.open(INDEXED_IMAGE_PATH)
indices = np.array(img)

flat = indices.flatten()
hist = np.bincount(flat, minlength=128)

total_pixels = len(flat)

used_indices = np.sum(hist > 0)
unused_indices = np.sum(hist == 0)

print("===== ANALIZA INDEKSÓW =====")
print(f"Liczba pikseli: {total_pixels:,}")
print(f"Użyte indeksy (0–127): {used_indices}")
print(f"Nieużyte indeksy: {unused_indices}")
print(f"Indeks 0 użyty pikseli: {hist[0]}")

print("\nNajczęściej używane indeksy:")
top_indices = np.argsort(hist)[-10:][::-1]
for idx in top_indices:
    print(f"Index {idx:3d}: {hist[idx]:,} pikseli")

print("\nIndeksy nieużywane:")
unused = np.where(hist == 0)[0]
print(unused)
