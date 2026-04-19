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
