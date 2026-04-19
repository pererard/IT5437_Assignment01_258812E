

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\MScAI_UOM\S3\CV\Assignment 01\data\runway.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
img_norm = img / 255.0  # normalize to [0,1]

gamma_05 = np.power(img_norm, 0.5)
gamma_05 = (gamma_05 * 255).astype(np.uint8)

gamma_2 = np.power(img_norm, 2)
gamma_2 = (gamma_2 * 255).astype(np.uint8)

r1, r2 = 0.2, 0.8

contrast = np.zeros_like(img_norm)

# Apply piecewise transformation
contrast[img_norm < r1] = 0
mask = (img_norm >= r1) & (img_norm <= r2)
contrast[mask] = (img_norm[mask] - r1) / (r2 - r1)
contrast[img_norm > r2] = 1

contrast = (contrast * 255).astype(np.uint8)

plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
plt.title("Gamma 0.5")
plt.imshow(gamma_05, cmap='gray')

plt.subplot(2,2,3)
plt.title("Gamma 2")
plt.imshow(gamma_2, cmap='gray')

plt.subplot(2,2,4)
plt.title("Contrast Stretching")
plt.imshow(contrast, cmap='gray')

plt.show()