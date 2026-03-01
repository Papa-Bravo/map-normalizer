import numpy as np
from PIL import Image
from skimage import color

ORIGINAL = "maps_raw/1020_00.tif"
INDEXED = "maps_indexed_v2/1020_00.tif"

print("Wczytywanie obrazów...")

orig = np.array(Image.open(ORIGINAL).convert("RGB"))
conv = np.array(Image.open(INDEXED).convert("RGB"))

assert orig.shape == conv.shape, "Rozmiary się nie zgadzają"

h, w, _ = orig.shape
pixels = h * w

print("Konwersja do LAB...")

orig_lab = color.rgb2lab(orig / 255.0)
conv_lab = color.rgb2lab(conv / 255.0)

print("Liczenie ΔE...")

delta_e = np.sqrt(np.sum((orig_lab - conv_lab) ** 2, axis=2))

mean_de = np.mean(delta_e)
max_de = np.max(delta_e)

pct_gt5 = np.sum(delta_e > 5) / pixels * 100
pct_gt10 = np.sum(delta_e > 10) / pixels * 100

print("\n===== RAPORT WALIDACJI =====\n")
print(f"Średni ΔE: {mean_de:.3f}")
print(f"Maksymalny ΔE: {max_de:.3f}")
print(f"% pikseli ΔE > 5:  {pct_gt5:.3f}%")
print(f"% pikseli ΔE > 10: {pct_gt10:.3f}%")
