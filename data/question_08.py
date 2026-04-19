import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread("D:\MScAI_UOM\S3\CV\Assignment 01\data\emma.jpg")

# Apply Gaussian sian smoothing
blurred = cv.GaussianBlur(im, (5, 5), 0)

# Apply median filtering
median = cv.medianBlur(im, 5)

# plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(im)
ax[0].set_title("Original Image")
ax[1].imshow(blurred)
ax[1].set_title("Gaussian Smoothing")
ax[2].imshow(median)
ax[2].set_title("Median Filtering")
plt.show()
