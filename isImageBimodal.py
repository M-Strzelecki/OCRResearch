import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def is_bimodal(image_path, peak_height_frac=0.2):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the histogram of the image
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # Compute the maximum peak height
    max_peak_height = max(hist)

    # Set the peak height parameter as a fraction of the maximum peak height
    peak_height = max_peak_height * peak_height_frac

    # Find the peaks of the histogram
    peaks, _ = find_peaks(hist, height=peak_height, distance=20, prominence=10)

    # If there are two peaks, the image is bimodal
    if len(peaks) == 2:
        return True, hist, peaks
    else:
        return False, hist, peaks


# Folder containing images
folder = './sample_images'

# Loop through all image files in the folder
for filename in os.listdir(folder):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Get the full path to the image file
        image_path = os.path.join(folder, filename)

        # Check if the image is bimodal
        bimodal, hist, peaks = is_bimodal(image_path)

        # Print the histogram and peaks
        plt.figure()
        plt.plot(hist)
        plt.plot(peaks, hist[peaks], "x")
        plt.title(f"Histogram for {filename}")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        plt.show()

        # Print the bimodal result
        if bimodal:
            print(f"{filename} is bimodal")
        else:
            print(f"{filename} is not bimodal")

