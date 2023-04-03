import numpy as np
import preprocessing as prep
import pytesseract
from pytesseract import Output
import cv2 as cv
import os

# importing image
image = cv.imread("./sample_images/nf157.jpg")
file_name = os.path.basename("./sample_images/nf157.jpg")
image_name = os.path.splitext(file_name)[0]


# resizing image and printing dimensions
image = cv.resize(image, (500, 500))
# print image after resizing
cv.imshow("Original Image", image)


# Apply gamma correction to image to reduce light bleeding
"""
gamma variable is set to 3, which represents the value of gamma used for gamma correction.
gamma_table is a NumPy array created using a list comprehension that applies the gamma correction 
    formula to each pixel value in the range of 0 to 255, and then scales it back to the range of 0 to 255. 
    This creates a lookup table for gamma correction.
The astype function is called to cast the pixel values to unsigned 8-bit integers, which is the expected data type for images in OpenCV.
The cv.LUT function is then used to apply the lookup table to the input image.
"""
gamma = 3
gamma_table = np.array(
    [((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]
).astype("uint8")
corrected_image = cv.LUT(image, gamma_table)

cv.imshow("Gamma = 3", corrected_image)
gamma2 = 0.3
gamma_table2 = np.array(
    [((i / 255.0) ** gamma2) * 255 for i in np.arange(0, 256)]
).astype("uint8")
corrected_image2 = cv.LUT(image, gamma_table2)
cv.imshow("Gamma = 0.3", corrected_image2)

cv.imshow("Original Image with Gamma", corrected_image)
# applying grayscale on gamma image
gray2 = prep.get_grayscale(corrected_image)
cv.imshow("Original Image with Gamma/Gray", gray2)
# applying thresholding
thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
cv.imshow("Original Image with Gamma/Gray/Thresh", thresh2)
# applying inverse to collor w+b -> b+w
thresh2_inv = cv.bitwise_not(thresh2)
cv.imshow("Original Image with Gamma/Gray/Thresh/Inverse", thresh2_inv)


# beforegray = cv.bitwise_not(image)
# cv.imshow("before gray", beforegray)


# converting image to grayscale
gray = prep.get_grayscale(image)
cv.imshow("Original Image with Gray", gray)


# Apply adaptive thresholding to binarize the image
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
cv.imshow("Original Image with Gray/Thresh ", thresh)
# applying inverse to collor w+b -> b+w
thresh_inv = cv.bitwise_not(thresh)
cv.imshow("Original Image with Gray/Thresh/Inverse", thresh_inv)


# details1 = prep.boundingboxes(agt)
# out1 = prep.returnoutput(details1)
# details2 = prep.boundingboxes(thresh)
# out2 = prep.returnoutput(details2)
# details3 = prep.boundingboxes(binarywithinvert)
# out3 = prep.returnoutput(details3)


# import re
#
# # Define the regular expression patterns
# calorie_pattern = re.compile(r'Calories[:\s]*(\d+)')
# fat_pattern = re.compile(r'Fat[:\s]*(\d+)')
# carb_pattern = re.compile(r'Carbohydrate[:\s]*(\d+)')
# protein_pattern = re.compile(r'Protein[:\s]*(\d+)')
#
# # Extract the nutritional information
# calories = calorie_pattern.search(text)
# fat = fat_pattern.search(text)
# carbs = carb_pattern.search(text)
# protein = protein_pattern.search(text)
#
# if calories:
#     calories = calories.group(1)
# else:
#     calories = 'N/A'
#
# if fat:
#     fat = fat.group(1)
# else:
#     fat = 'N/A'
#
# if carbs:
#     carbs = carbs.group(1)
# else:
#     carbs = 'N/A'
#
# if protein:
#     protein = protein.group(1)
# else:
#     protein = 'N/A'
#
#
# print(f'Calories: {calories}')
# print(f'Fat: {fat}')
# print(f'Carbs: {carbs}')
# print(f'Protein: {protein}')

# pulling information from ocr processed images that used different preprocesses
text = pytesseract.image_to_string(thresh2)
print("Gamma/Gray/Thresh(GGT)=:", text)

text2 = pytesseract.image_to_string(thresh2_inv)
print("Gamma/Gray/Thresh/Inverse(GGTI)=:", text2)

text3 = pytesseract.image_to_string(thresh)
print("Gray/Thresh(GT)=:", text3)

text4 = pytesseract.image_to_string(thresh_inv)
print("Gray/Thresh/Inverse(GTI)=:", text4)


# using function from preprocessing.py to get character count from different ocr outputs
character_count = prep.count_characters(text)
print(f"Total number of characters(GGT): {character_count}")
character_count2 = prep.count_characters(text2)
print(f"Total number of characters(GGTI): {character_count2}")
character_count3 = prep.count_characters(text3)
print(f"Total number of characters(GT): {character_count3}")
character_count4 = prep.count_characters(text4)
print(f"Total number of characters(GTI): {character_count4}")

cv.waitKey(0)
cv.destroyAllWindows()
