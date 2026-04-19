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