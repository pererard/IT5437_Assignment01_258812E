import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

size = 5
sigma = 2

ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel = kernel / np.sum(kernel)  # normalize Gaussian

kernel_x = -(xx / sigma**2) * kernel
kernel_y = -(yy / sigma**2) * kernel

# normalize
kernel_x = kernel_x / np.sum(np.abs(kernel_x))  # detects vertical edges
kernel_y = kernel_y / np.sum(np.abs(kernel_y))  # detects horizontal edges


size = 51
sigma = 2  # standard deviation

# Coordinate grid
ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

# Gaussian
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= np.sum(kernel)  # normalize Gaussian

# Derivative kernels
kernel_x = -(xx / sigma**2) * kernel
kernel_y = -(yy / sigma**2) * kernel

fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot_surface(xx, yy, kernel_x, cmap="viridis")

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Kernel Value")
ax3d.set_title("Derivative-of-Gaussian Kernel (X-direction)")

plt.show()

im = cv.imread("D:\MScAI_UOM\S3\CV\Assignment 01\data\emma.jpg", cv.IMREAD_GRAYSCALE)

# image height and width
height, width = im.shape

size = 51
pad = size // 2

# Pad image to handle edges
padded = np.pad(im, pad, mode="reflect")  # reflect is better than zero padding

# Initialize outputs
Ix = np.zeros_like(im, dtype=float)
Iy = np.zeros_like(im, dtype=float)

# Manual convolution
for i in range(height):
    for j in range(width):
        region = padded[i : i + size, j : j + size]
        Ix[i, j] = np.sum(region * kernel_x)
        Iy[i, j] = np.sum(region * kernel_y)

gradient_magnitude = np.sqrt(Ix**2 + Iy**2)

# Normalize for display
gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(
    np.uint8
)


plt.figure(figsize=(15, 5))

# Horizontal gradient
plt.subplot(1, 3, 1)
plt.title("Horizontal Gradient (Ix - kernel_x)")
plt.imshow(Ix, cmap="gray")
plt.axis("off")

# Vertical gradient
plt.subplot(1, 3, 2)
plt.title("Vertical Gradient (Iy - kernel_y)")
plt.imshow(Iy, cmap="gray")
plt.axis("off")

# Gradient magnitude
plt.subplot(1, 3, 3)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap="gray")
plt.axis("off")

plt.show()