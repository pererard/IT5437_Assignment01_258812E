import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def bilateral_filter(gray, d, sigma_s, sigma_r):
    # dimensions
    h, w = gray.shape

    # Add Padding
    pad = d // 2
    padded = np.pad(gray, pad, mode="reflect")

    output = np.zeros_like(gray, dtype=np.float32)

    # spatial Gaussian
    spatial_kernel = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            x = i - pad
            y = j - pad
            spatial_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # Apply filter
    for i in range(h):
        for j in range(w):
            center = padded[i + pad, j + pad]

            wp = 0.0  # normalization factor
            filtered = 0.0

            for ki in range(d):
                for kj in range(d):
                    neighbor = padded[i + ki, j + kj]

                    # Intensity difference
                    range_weight = np.exp(
                        -((neighbor - center) ** 2) / (2 * sigma_r**2)
                    )

                    # Combined weight
                    weight = spatial_kernel[ki, kj] * range_weight

                    filtered += neighbor * weight
                    wp += weight

            output[i, j] = filtered / wp

    return output.astype(np.uint8)


im = cv.imread(r"D:\MScAI_UOM\S3\CV\Assignment 01\data\emma.jpg")
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

filtered = bilateral_filter(gray, d=5, sigma_s=75, sigma_r=75)


# Gaussian smoothing using OpenCV
im_gaussian = cv.GaussianBlur(gray, (5, 5), 75)

cv_bilateral = cv.bilateralFilter(gray, 5, 75, 75)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(im_gaussian, cmap="gray")
axes[1].set_title("Gaussian Blur (OpenCV)")
axes[1].axis("off")

axes[2].imshow(cv_bilateral, cmap="gray")
axes[2].set_title("Bilateral Filter (OpenCV)")
axes[2].axis("off")

axes[3].imshow(filtered, cmap="gray")
axes[3].set_title("Bilateral Filter (Manual)")
axes[3].axis("off")

plt.tight_layout()
plt.show()
