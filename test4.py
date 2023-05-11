import cv2
import pytesseract
import re
from collections import Counter
import preprocessing as prep
import numpy as np
import csv
import os

# Load image
# filename ="./sample_images/nf97.jpg"
filename = "./sample_images/nf131.jpg"
image = cv2.imread(filename)
filename_string = os.path.basename(filename)
filename_s = filename_string
filename_string = filename_string.split(".")[0]
print(filename_string)
image = cv2.resize(image, (400, 400))
# image = prep.resize_image(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the range of values to test for blockSize and C 
blockSizes = range(11, 61, 2) # odd numbers
Cs = range(1, 30)

# Open the CSV file and read its contents
with open('hardfulltext.csv', 'r') as file:
    reader = csv.reader(file)
    # Loop over each row in the CSV file
    for row in reader:
        # If the file name in the row matches the image name, extract the results
        if row[0] == filename_s:
            full_text = row[1]
            full_text = re.sub("[^a-z0-9.]", "", full_text.lower())
            break
# Use the extracted results if available
if 'full_text' in locals():
    full_text_count = prep.count_chars(full_text)    
    dict_full_text = prep.string_to_dict(full_text_count)
else:
    print('No results found for file:', filename_string)

# Initialize variables to store the best parameters and performance
bestBlockSize = None
bestC = None
bestPerformance = float('inf')
# Define the hard-coded performance value and tolerance
targetPerformance = 1
tolerance = 0.1
performance_threshold = min(targetPerformance * 1.2 , 100)
# Loop over all possible combinations of blockSize and C
for blockSize in blockSizes:
    for C in Cs:
        print("Testing blockSize:", blockSize, "C:", C)
        
        # Apply adaptive thresholding with the current parameter values
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
        text = pytesseract.image_to_string(thresh)
        # Convert thresholding results to text and convert them to dict to calculate SMAPE performance
        performanced = pytesseract.image_to_string(thresh)
        performanced = re.sub("[^a-z0-9.]", "", text.lower())
        performanced = prep.count_chars(text)
        performanced = prep.string_to_dict(performanced)
        performance = 0
        for key in dict_full_text.keys():
            a = dict_full_text[key]
            b = performanced.get(key, 0)
            performance += abs(a - b) / (a + b)
        performance = (performance / len(dict_full_text)) * 100
        print("SMAPE: {:.2f}%".format(performance))
        
        # If the current parameter values produce a better performance update the best parameters and performance
        if performance < bestPerformance:
            bestBlockSize = blockSize
            print("Best Block Size: ",bestBlockSize)
            bestC = C
            print("Best C: ", bestC)
            bestPerformance = performance
            print('Best Performance: ', bestPerformance)
            # Apply adaptive thresholding with the new best parameters to check if it meets the criteria
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bestBlockSize, bestC)

# Apply adaptive thresholding with the best parameters
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bestBlockSize, bestC)
text = pytesseract.image_to_string(thresh)
# text = re.sub("[^a-z0-9.]", "", text.lower())

# Display the original and thresholded images side by side
cv2.imshow('Original', image)
cv2.imshow('Thresholded', thresh)

# Print the best parameters and performance
print('Best blockSize:', bestBlockSize)
print('Best C:', bestC)
print('Best performance:', bestPerformance)
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()