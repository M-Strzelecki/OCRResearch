import cv2
import pytesseract
import re
from collections import Counter
import preprocessing as prep
import numpy as np
import csv
import os

images_folder = "./sample_images/"
output_csv_file = 'outputautogamma.csv'
blockSizes = range(11, 21, 2)
Cs = range(1, 5)

with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'best_block_size', 'best_c', 'best_performance'])
    for filename in os.listdir(images_folder):
        print ('Filename: %s' % filename)
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(images_folder, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dict_full_text = {}
            full_text = ""
            with open('hardfulltext.csv', 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if row[0] == filename:
                        full_text = row[1]
                        full_text = re.sub("[^a-z0-9.]", "", full_text.lower())
                        break
            if full_text:
                full_text_count = {}
                for char in full_text:
                    full_text_count[char] = full_text_count.get(char, 0) + 1
                dict_full_text = full_text_count
            best_block_size = None
            best_c = None
            best_performance = float('inf')
            target_performance = 1
            tolerance = 0.1
            performance_threshold = min(target_performance * 1.2 , 100)
            for block_size in blockSizes:
                for c in Cs:
                    print("Block size: " + str(block_size))
                    print("c = " + str(c))
                    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
                    text = pytesseract.image_to_string(thresh)
                    text = re.sub("[^a-z0-9.]", "", text.lower())
                    text_count = {}
                    for char in text:
                        text_count[char] = text_count.get(char, 0) + 1
                    performance = 0
                    for key in dict_full_text.keys():
                        a = dict_full_text[key]
                        b = text_count.get(key, 0)
                        performance += abs(a - b) / (a + b)
                    performance = (performance / len(dict_full_text)) * 100
                    if performance < best_performance:
                        best_block_size = block_size
                        best_c = c
                        best_performance = performance
                        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, best_block_size, best_c)
            writer.writerow([filename, best_block_size, best_c, best_performance])