import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

im = cv.imread("D:\MScAI_UOM\S3\CV\Assignment 01\data\Figure_3.jpg")

# convert to gray scale
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, binary_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


# 'ret' is the threshold value chosen by Otsu
print("Otsu's threshold value:", ret)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Grayscale Image")
axes[0].axis("off")

axes[1].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
axes[1].axvline(x=ret, color="red", linewidth=2, label=f"Otsu threshold = {int(ret)}")
axes[1].set_title("Grayscale Histogram")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
axes[1].legend()

axes[2].imshow(binary_mask, cmap="gray")
axes[2].set_title("Binary Mask (Otsu)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
