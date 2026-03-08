import numpy as np
from skimage import color

palette_rgb = np.load("palette/palette_v3_127_rgb.npy")
palette_lab = color.rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0)[0]

reference_colors = [
    (0, 0, 0),
    (255, 255, 255),
    (150, 200, 240),
    (200, 230, 250),
    (0, 90, 170),
    (242, 234, 199),
    (220, 0, 0),
    (0, 160, 0),
    (255, 220, 0),
    (160, 160, 160),
    (200, 0, 120),
    (80, 140, 80),
]

print("\n===== TEST REFERENCYJNYCH (v3) =====\n")

for i, rgb in enumerate(reference_colors, start=1):
    ref_lab = color.rgb2lab(np.array(rgb).reshape(1,1,3)/255.0)[0][0]
    pal_lab = palette_lab[i]
    delta_e = np.sqrt(np.sum((pal_lab - ref_lab)**2))
    print(f"Index {i:3d}  RGB={rgb}  ΔE={delta_e:.4f}")
