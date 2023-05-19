import cv2
import pytesseract
import re
from collections import Counter
import preprocessing as prep
import numpy as np
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Define input folder with images
images_folder = "./sample_images/"
# Define csv file name to store results
output_csv_file = './heatmapdata/heatmap_data.csv'

# Define the range of values to test for blockSize and C 
blockSizes = range(11, 61, 2) # odd numbers, minimum starting range = 11
Cs = range(0, 31, 2)
# blockSizes = range(11, 21, 2) # odd numbers
# Cs = range(0, 6, 2)

# Initialize variables to store the best parameters and performance
bestBlockSize = None
bestC = None
bestPerformance = float('inf')

results = []

final_results = []
x_values = []
y_values = []

# Loop over all possible combinations of blockSize and C
for blockSize in blockSizes:
    for C in Cs:
        print("Testing blockSize:", blockSize, "C:", C)
        # Variable to store the performance values(smape)
        smape_avg = []
        for filename in os.listdir(images_folder):
            print ('Filename: %s' % filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load the image and preprocess it.
                image_file = os.path.join(images_folder, filename)
                with open('./hardcodednutrilabels/hardfulltext.csv', 'r') as file:
                    reader = csv.reader(file)
                    # Loop over each row in the CSV file
                    for row in reader:
                        # If the file name in the row matches the image name, extract the results
                        if row[0] == filename:
                            full_text = row[1]
                            full_text = re.sub("[^a-z0-9.]", "", full_text.lower())
                            break
                # Use the extracted results if available
                if 'full_text' in locals():
                    full_text_count = prep.count_chars(full_text)    
                    dict_full_text = prep.string_to_dict(full_text_count)
                else:
                    print('No results found for file:', filename)

                image = cv2.imread(image_file)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                smape_avg.append(performance)
                print("SMAPE: {:.2f}%".format(performance))
        
        x_values.append(blockSize)
        y_values.append(C)
        bestBlockSize = blockSize
        print("Best Block Size: ",bestBlockSize)
        bestC = C
        print("Best C: ", bestC)
        # Calculate Average SMAPE Score
        total_result = sum(smape_avg)
        total_result_elems = len(smape_avg)
        total_result_avg_score = total_result / total_result_elems
        print(f"Total Accuracy Average: {total_result_avg_score:.2f}%")
        results.append((blockSize, C, total_result_avg_score))
        final_results.append((bestBlockSize, bestC, total_result_avg_score))

        df = pd.DataFrame(final_results, columns=['blockSize', 'C', 'total_result_avg_score'])
        # final_results.append({'blockSize': bestBlockSize, 'C': bestC, 'score': total_result_avg_score})

        # Create a Pandas DataFrame from the list of dictionaries
        # df = pd.DataFrame(final_results)

        # Write the DataFrame to a CSV file
        # df.to_csv('thresholding_results2.csv', index=False)

# Create a Pandas dataframe from the results list
df = pd.DataFrame(results, columns=['blockSize', 'C', 'total_result_avg_score'])

# Pivot the dataframe to create a matrix of total_result_avg_score values
heatmap_data = df.pivot(index='C', columns='blockSize', values='total_result_avg_score')
# heatmap_data.to_csv('heatmap_data2.csv')
heatmap_data.to_csv(output_csv_file)
# Create the heatmap using seaborn
sns.set(font_scale=1.2)
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, cbar_kws={'label': 'Total Result Average Score'})
plt.xlabel('Block Size')
plt.ylabel('C')
plt.title('SMAPE Scores')
plt.show()