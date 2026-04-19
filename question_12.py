import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\MScAI_UOM\S3\CV\Assignment 01\data\emma.jpg', 0)
img = img / 255.0

# Step 1: Log transform
log_img = np.log1p(img)

# Step 2: FFT
fft = np.fft.fft2(log_img)
fft_shift = np.fft.fftshift(fft)

# Step 3: Create High-pass filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2

H = np.ones((rows, cols))
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        H[i,j] = 1 - np.exp(-(d**2)/(2*(30**2)))  # cutoff freq = 30

# Step 4: Apply filter
filtered = fft_shift * H

# Step 5: Inverse FFT
ifft_shift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(ifft_shift)
img_back = np.real(img_back)

# Step 6: Exponential
exp_img = np.expm1(img_back)

# Normalize
exp_img = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
result = exp_img.astype(np.uint8)

plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Log Image
plt.subplot(2,3,2)
plt.title("Log Transformed")
plt.imshow(log_img, cmap='gray')
plt.axis('off')

# FFT Magnitude
plt.subplot(2,3,3)
plt.title("FFT Spectrum")
plt.imshow(np.log(1 + np.abs(fft_shift)), cmap='gray')
plt.axis('off')

# Filter
plt.subplot(2,3,4)
plt.title("High-pass Filter")
plt.imshow(H, cmap='gray')
plt.axis('off')

# Filtered Spectrum
plt.subplot(2,3,5)
plt.title("Filtered Spectrum")
plt.imshow(np.log(1 + np.abs(filtered)), cmap='gray')
plt.axis('off')

# Final Result
plt.subplot(2,3,6)
plt.title("Final Result")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()