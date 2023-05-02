import cv2
import pytesseract
import re
from collections import Counter
import preprocessing as prep
import numpy as np
import csv
import os

# Load image
filename ="./sample_images/nf97.jpg"
# filename = "./sample_images/nf131.jpg"
image = cv2.imread(filename)
filename_string = os.path.basename(filename)
filename_string = filename_string.split(".")[0]
print(filename_string)
image = cv2.resize(image, (400, 400))
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Define the range of values to test for blockSize and C (1-100)
blockSizes = range(11, 61, 2) # odd numbers between 1 and 100
Cs = range(1, 16)


# Open the CSV file and read its contents
with open('hard_nutri.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Loop over each row in the CSV file
    for row in reader:
        # If the file name in the row matches the image name, extract the results
        if row[0] == filename_string:
            Character_Count = row[1]
            # ... extract other results as needed
            break  # Stop searching once a match is found

# Use the extracted results as needed
print('Hard Character Count:', Character_Count)


# Initialize variables to store the best parameters and performance
bestBlockSize = None
bestC = None
bestPerformance = 0
# Define the hard-coded performance value and tolerance
targetPerformance = int(Character_Count)
tolerance = 0.1
# Loop over all possible combinations of blockSize and C
for blockSize in blockSizes:
    for C in Cs:
        print("Testing blockSize:", blockSize, "C:", C)
        
        # Apply adaptive thresholding with the current parameter values
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
        text = pytesseract.image_to_string(thresh)
        # Compute the performance metric (e.g., accuracy, F1 score, etc.) for the thresholded image
        # In this example, we will use the mean pixel value of the thresholded image as the performance metric
        performance = prep.count_characters(text)
        print("Performance: ", performance)
        
        # If the current parameter values produce a better performance, update the best parameters and performance
        if performance > bestPerformance and performance >= targetPerformance * (1 - tolerance):
            print('tp', targetPerformance)
            bestBlockSize = blockSize
            bestC = C
            bestPerformance = performance
            # Apply adaptive thresholding with the new best parameters to check if it meets the criteria
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bestBlockSize, bestC)
            performance = prep.count_characters(text)
            if performance >= targetPerformance * (1 - tolerance):
                break
    # If the performance is within 10% of the target performance, stop the loop and apply adaptive thresholding with the best parameters
    if performance >= targetPerformance * (1 - tolerance):
        break        
# Apply adaptive thresholding with the best parameters
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bestBlockSize, bestC)

text = pytesseract.image_to_string(thresh)
print(text.lower())

text2 = re.sub("[^a-z0-9.]", "", text.lower())
print(text2)
counter = Counter(text2)
print(counter)

# char_counts = prep.count_chars(text)
# print(char_counts)
# Display the original and thresholded images side by side
cv2.imshow('Original', image)
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Print the best parameters and performance
print('Best blockSize:', bestBlockSize)
print('Best C:', bestC)
print('Best performance:', bestPerformance)
print('Target Performance:', targetPerformance)