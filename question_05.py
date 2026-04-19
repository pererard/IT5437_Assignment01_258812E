import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


size = 5
sigma = 2

# coordinate grid
ax = np.arange(-(size // 2), (size // 2 + 1))  # array with values from -2 to 2
print(ax)
xx, yy = np.meshgrid(ax, ax)


# Gausian kernel
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
print("Raw kernel: ", kernel)

# normzalize the kernel
kernel = kernel / np.sum(kernel)
print("Normalized kernel: ", kernel)


# 51x51 kernel
size = 51
sigma = 10  # random value

ax = np.arange(-(size // 2), size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)

kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= np.sum(kernel)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(xx, yy, kernel)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Kernel Value")

plt.show()

