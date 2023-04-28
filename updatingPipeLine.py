import pytesseract
import cv2
import numpy as np
import preprocessing as prep
import pandas as pd
import re
from collections import Counter

image = cv2.imread("./sample_images/nf90.jpg")
image = cv2.resize(image, (400, 400))

gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

text = pytesseract.image_to_string(thresh)
print(text.lower())

text2 = re.sub("[^a-z0-9.]", "", text.lower())
print(text2)
counter = Counter(text2)
print(counter)

char_counts = prep.count_chars(text)
print(char_counts)

import re

# Define regex patterns to match the nutritional information to extract
calorie_pattern = re.compile(r"Calories[:\s]*(\d+)")
fat_pattern = re.compile(r"Fat[:\s]*(\d+)")
carb_pattern = re.compile(r"Carbohydrates[:\s]*(\d+)")
protein_pattern = re.compile(r"Protein[:\s]*(\d+)")

# Search for the first occurrence of each pattern in the OCR output text.
calories = calorie_pattern.search(text)
fat = fat_pattern.search(text)
carbs = carb_pattern.search(text)
protein = protein_pattern.search(text)

# Extract the numerical values that match each pattern, or assign 'N/A' if no match is found
if calories:
    calories = calories.group(1)
else:
    calories = "N/A"

if fat:
    fat = fat.group(1)
else:
    fat = "N/A"

if carbs:
    carbs = carbs.group(1)
else:
    carbs = "N/A"

if protein:
    protein = protein.group(1)
else:
    protein = "N/A"


print(f"Calories: {calories}")
print(f"Fat: {fat}")
print(f"Carbs: {carbs}")
print(f"Protein: {protein}")

character_count = prep.count_characters(text)
print(f"Total number of characters: {character_count}")


input_folder = "./sample_images"
output_file = "output_v1_3.csv"
full_text = "fulltext_v1_3.csv"
hard = pd.read_csv("hardfulltext.csv")

prep.process_images(input_folder, output_file)
prep.process_images_individual(input_folder, full_text)
