import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


f = cv.imread("D:\MScAI_UOM\S3\CV\Assignment 01\data\jeniffer.jpg", cv.IMREAD_GRAYSCALE)
f = f.astype(np.float32)

# Simple 5x5 averaging filter
h = np.ones((5, 5), dtype=np.float32) / 25

# Spatial domain convolution
g_spatial = convolve2d(f, h, mode="same", boundary="wrap")

# Frequency domain multiplication
rows, cols = f.shape

# Pad to image size
H_padded = np.zeros((rows, cols), dtype=np.float32)

kh, kw = h.shape
center_r, center_c = kh // 2, kw // 2

# Place kernel centered at (0,0) for FFT
H_padded[:kh, :kw] = h
H_padded = np.roll(H_padded, -center_r, axis=0)
H_padded = np.roll(H_padded, -center_c, axis=1)

# FFT
F = np.fft.fft2(f)
H = np.fft.fft2(H_padded)

# Multiply in frequency domain
G = F * H

# Inverse FFT
g_freq = np.fft.ifft2(G)
g_freq = np.real(g_freq)

difference = np.abs(g_spatial - g_freq)

print("Max difference:", np.max(difference))


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Spatial Convolution")
plt.imshow(g_spatial, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Frequency Domain")
plt.imshow(g_freq, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(difference, cmap="gray")

plt.show()