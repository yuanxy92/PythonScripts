import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load image
image_3ch = cv2.imread('div_000000.png')  # your metalens image
correction_matrices = []

for ch in range(3):
    image = image_3ch[:, :, ch]
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    gray = gray.astype(np.float32) / 255.0

    # Get image center
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    cy = cy + 15
    # cx = cx - 10

    # --- Compute radial distance map ---
    y_grid, x_grid = np.indices((h, w))
    r_grid = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    r_norm = r_grid / np.max(r_grid)

    # --- Estimate expected brightness falloff roughly ---
    # Assume center is brightest; take a rough max in central region
    central_mask = r_norm < 0.1
    central_brightness = np.median(gray[central_mask])

    # Create a naive expected brightness profile (monotonically decreasing)
    # You can fit a better one iteratively if needed
    expected_profile = lambda r: central_brightness * (1 - 0.5 * r**2)  # Falloff curve

    # --- Create spatially varying threshold ---
    adaptive_threshold = expected_profile(r_norm)

    # Mask where pixels are relatively close to expected brightness (i.e. "white")
    white_mask = gray > (adaptive_threshold * 0.9)  # 0.9 = tolerance factor

    # Optional: Clean the mask (remove noise)
    white_mask_clean = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                        kernel=np.ones((3, 3), np.uint8))
    # Erosion to get confident inner white areas
    kernel = np.ones((25, 25), np.uint8)
    white_mask_clean = cv2.erode(white_mask_clean, kernel, iterations=1)

    # --- Visualize the mask ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Input (Gray)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(adaptive_threshold, cmap='gray')
    plt.title('Adaptive Threshold Map')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(white_mask_clean, cmap='gray')
    plt.title('White Pixel Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    # Coordinates of white pixels
    ys, xs = np.where(white_mask_clean > 0)

    # Compute distances from center
    rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    intensities = gray[ys, xs]

    # Normalize distance
    rs_norm = rs / np.max(rs)

    # Fit a polynomial curve to intensity vs. distance
    # def vignette_model(r, a, b, c):
    #     return a * r**2 + b * r + c
    # def vignette_model(r, a, b, c, d):
    #     return a * r**4 + b * r**2 + c * r + d
    def vignette_model(r, a, b, c, d, e):
        return a * r**4 + b * r**3 + c * r**2 + d * r + e

    params, _ = curve_fit(vignette_model, rs_norm, intensities)
    print("Fitted parameters:", params)

    # Create full-size correction map
    y_grid, x_grid = np.indices((h, w))
    r_grid = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    r_norm = r_grid / np.max(r_grid)

    # Compute correction factor
    vignette_map = vignette_model(r_norm, *params)
    correction_map = 1.0 / vignette_map
    correction_map = correction_map / np.max(correction_map) * 2 # Normalize

    # Apply correction to each channel
    image_float = image.astype(np.float32) / 255.0
    corrected = np.clip(image_float * correction_map, 0, 1)

    correction_matrices.append(correction_map)

    # Save or display
    # cv2.imwrite('corrected_image.jpg', (corrected * 255).astype(np.uint8))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow((corrected * 255).astype(np.uint8))
    plt.title("Vignette Corrected")
    plt.show()

    # Plot intensity vs. distance
    plt.figure()
    plt.scatter(rs_norm, intensities, s=2, label='Samples')
    r_fit = np.linspace(0, 1, 100)
    i_fit = vignette_model(r_fit, *params)
    plt.plot(r_fit, i_fit, color='red', label='Fitted curve')
    plt.xlabel("Normalized Radius")
    plt.ylabel("Intensity")
    plt.title("Vignetting Profile")
    plt.legend()
    plt.show()

correction_matrix = np.stack(correction_matrices, axis=0)
np.save('correction_matrix.npy', correction_matrix)