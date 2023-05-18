import cv2
import numpy as np
from matplotlib import pyplot as plt
"""
Just a code used to generate some examples of different gamma levels for presentation
"""
# Load the image
img = cv2.imread("./sample_images/nf131.jpg")
img = cv2.resize(img, (400, 400))

# Define the gamma value
gamma = 1.5
gamma2 = 0.2
gamma3 = 2
# Generate the lookup table
table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
    "uint8"
)
table2 = np.array([((i / 255.0) ** gamma2) * 255 for i in np.arange(0, 256)]).astype(
    "uint8"
)
table3 = np.array([((i / 255.0) ** gamma3) * 255 for i in np.arange(0, 256)]).astype(
    "uint8"
)

# Apply the gamma correction using the lookup table
img_gamma_corrected = cv2.LUT(img, table)
img_gamma_corrected2 = cv2.LUT(img, table2)
img_gamma_corrected3 = cv2.LUT(img, table3)
# Plot the original and gamma-corrected image histograms
plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 4, 2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("Original Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2RGB))
plt.title("Gamma-Corrected Image, G = 1.5")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 4, 4)
plt.hist(img_gamma_corrected.ravel(), 256, [0, 256])
plt.title("Gamma-Corrected Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(img_gamma_corrected2, cv2.COLOR_BGR2RGB))
plt.title("Gamma-Corrected Image, G = 0.2")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 4, 6)
plt.hist(img_gamma_corrected2.ravel(), 256, [0, 256])
plt.title("Gamma-Corrected Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(img_gamma_corrected3, cv2.COLOR_BGR2RGB))
plt.title("Gamma-Corrected Image, G = 2")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 4, 8)
plt.hist(img_gamma_corrected3.ravel(), 256, [0, 256])
plt.title("Gamma-Corrected Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.tight_layout()
plt.show()
