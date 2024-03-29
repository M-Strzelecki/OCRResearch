import re
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output
import csv
import os
from itertools import zip_longest
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


# function to convert image to grayscale
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def boundingboxes(image):
    # Define custom Tesseract configuration for recognizing text
    custom_config = r"--oem 3 --psm 11"

    # Use Tesseract to get text data from the input image
    details = pytesseract.image_to_data(
        image, output_type=Output.DICT, config=custom_config, lang="eng"
    )

    # Print keys of the dictionary returned by Tesseract
    print(details.keys())

    # Get total number of text boxes detected by Tesseract
    total_boxes = len(details["text"])

    # Loop through each text box and draw a bounding box around it if the confidence level is high enough
    for sequence_number in range(total_boxes):
        if (
            int(details["conf"][sequence_number]) > 30
        ):  # set minimum confidence level to 30
            # Extract the bounding box coordinates from the dictionary returned by Tesseract
            (x, y, w, h) = (
                details["left"][sequence_number],
                details["top"][sequence_number],
                details["width"][sequence_number],
                details["height"][sequence_number],
            )
            # Draw a rectangle around the text box on the input image using OpenCV
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes drawn
    cv.imshow("captured data", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Return the dictionary containing text data and bounding box coordinates
    return details


def returnoutput(details):
    # Initialize empty list to hold the parsed text
    parse_text = []

    # Initialize empty list to hold words within each line of text
    word_list = []

    # Initialize variable to hold the last word seen
    last_word = ""

    # Iterate through each word in the text of the details dictionary
    for word in details["text"]:
        # If the current word is not an empty string, append it to the current line of text
        if word != "":
            word_list.append(word)

            # Update the last word seen to be the current word
            last_word = word

        # If the current word is an empty string and the last word seen was not empty,
        # or if the current word is the last word in the text, then the current line of text is complete
        # and can be added to the parsed text list
        if (last_word != "" and word == "") or (word == details["text"][-1]):
            parse_text.append(word_list)

            # Reset the word list to empty for the next line of text
            word_list = []

    # Print the parsed text list
    print(parse_text)


"-------------------------------------------------------------------"

"------------------EDIT FOR updatingPipeLine------------------------"
def preprocess_image(image):
    """
    Preprocesses an input image for OCR.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The preprocessed image as a binary NumPy array.
    """

    
    # Resize the image to a fixed size of 400x400 pixels
    image = cv.resize(image, (400, 400))

    gray = adaptive_gamma_correction(image)
    # Apply Binary Inverse thresholding to the grayscale image to obtain a binary image.
    thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)[1]

    # Return the preprocessed binary image.
    return thresh


def extract_nutrition_info(text):
    """
    Extracting the nutritional information from the OCR output text using regular expressions (regex) to match and
    extract specific text from a string.

    Args:
        text (str): The OCR output text.

    Returns:
        tuple: A tuple containing the extracted nutritional information (calories, fat, carbs, protein).
    """

    # Define regex patterns to match the nutritional information to extract.
    calorie_pattern = re.compile(r"Calories[:\s]*(\d+)")
    fat_pattern = re.compile(r"Fat[:\s]*(\d+)")
    carb_pattern = re.compile(r"Carbohydrate[:\s]*(\d+)")
    protein_pattern = re.compile(r"Protein[:\s]*(\d+)")

    # Search for the first occurrence of each pattern in the OCR output text.
    calories = calorie_pattern.search(text)
    fat = fat_pattern.search(text)
    carbs = carb_pattern.search(text)
    protein = protein_pattern.search(text)

    # Extract the numerical values that match each pattern, or assign 'N/A' if no match is found.
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

    # Return the extracted nutritional information as a tuple.
    return calories, fat, carbs, protein


def count_characters(ocr_output):
    """
    Counts the total number of characters in the OCR output text.

    Args:
        ocr_output (str): The OCR output text.

    Returns:
        int: The total number of characters in the OCR output text.
    """
    return len(ocr_output)


def process_images(input_folder, output_file):
    """
    Processes all images in a folder, extracts nutritional information from them using OCR, and saves the information to a CSV file.

    Args:
        input_folder (str): The path to the folder containing the input images.
        output_file (str): The path to the CSV file to which the nutritional information will be saved.
    """

    # Open the output CSV file and create a CSV writer object.
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file.
        writer.writerow(
            ["id", "Character Count", "Calories", "Fat", "Carbs", "Protein"]
        )

        # Process each file in the input folder.
        for filename in os.listdir(input_folder):
            # Only process files with the '.jpg' or '.png' extension.
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load the image and preprocess it.
                image_file = os.path.join(input_folder, filename)
                image = cv.imread(image_file)
                processed_image = preprocess_image(image)

                # Perform OCR on the preprocessed image and extract the nutritional information.
                ocr_output = pytesseract.image_to_string(processed_image)
                calories, fat, carbs, protein = extract_nutrition_info(ocr_output)
                character_count = count_characters(ocr_output)

                # Write the extracted nutritional information and character count to the CSV file.
                writer.writerow(
                    [filename, character_count, calories, fat, carbs, protein]
                )


def process_images_individual(input_folder, output_file):
    """
    Processes all images in a folder, extracts nutritional information from them using OCR, and saves the information to a CSV file.

    Args:
        input_folder (str): The path to the folder containing the input images.
        output_file (str): The path to the CSV file to which the nutritional information will be saved.
    """

    # Open the output CSV file and create a CSV writer object.
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file.
        writer.writerow(["id", "full text"])

        # Process each file in the input folder.
        for filename in os.listdir(input_folder):
            # Only process files with the '.jpg' or '.png' extension.
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load the image and preprocess it.
                image_file = os.path.join(input_folder, filename)
                image = cv.imread(image_file)
                processed_image = preprocess_image(image)

                # Perform OCR on the preprocessed image and extract the nutritional information.
                ocr_output = pytesseract.image_to_string(processed_image)
                pattern = r"\s+"
                ocr_output = re.sub(pattern, "", ocr_output)

                # Write the extracted nutritional information and character count to the CSV file.
                writer.writerow([filename, ocr_output])


"---------------------Compare CSV---------------------------------"


def read_csv_file(filename):
    """
    Reads a CSV file and returns the data as a list of dictionaries.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: The data from the CSV file as a list of dictionaries.
    """
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        # Sort the rows by the "id" column.
        sorted_rows = sorted(reader, key=lambda row: row["id"])
        return list(sorted_rows)


def compare_csv_files(file1, file2):
    """
    Compares the character counts in two CSV files and returns a list of dictionaries representing the results.

    Args:
        file1 (str): The path to the first CSV file.
        file2 (str): The path to the second CSV file.

    Returns:
        list: A list of dictionaries representing the comparison results.
    """
    # Read the rows from both CSV files.
    data1 = read_csv_file(file1)
    data2 = read_csv_file(file2)

    # Sort both datasets by the 'id' column.
    data1 = sorted(data1, key=lambda x: x["id"])
    data2 = sorted(data2, key=lambda x: x["id"])

    # Compare the character counts for each row in both datasets.
    results = []
    result_avg = []
    for row1, row2 in zip(data1, data2):
        accuracy = None
        if row1["Character Count"] != "N/A" and row2["Character Count"] != "N/A":
            count1 = int(row1["Character Count"])
            count2 = int(row2["Character Count"])
            if count1 == count2:
                accuracy = 100
            else:
                accuracy = (min(count1, count2) / max(count1, count2)) * 100
        results.append(
            {
                "id": row1["id"],
                "file1": file1,
                "file2": file2,
                "count1": row1["Character Count"],
                "count2": row2["Character Count"],
                "accuracy": accuracy,
            }
        )
        result_avg.append(accuracy)
    total_result = sum(result_avg)
    total_result_elems = len(result_avg)
    total_result_avg_score = total_result / total_result_elems
    print(f"Total Accuracy Average: {total_result_avg_score:.2f}%")


    return results


def print_comparison_results(results):
    """
    Prints the comparison results to the console.

    Args:
        results (list): A list of dictionaries representing the comparison results.
    """
    for result in results:
        print(f"ID: {result['id']}")
        print(f"File 1 ({result['file1']}): {result['count1']}")
        print(f"File 2 ({result['file2']}): {result['count2']}")
        if result["accuracy"] is not None:
            print(f"Accuracy: {result['accuracy']:.2f}%")
        else:
            print("Accuracy: N/A")
        print()


"----------------------------Count Individual Characters-----------------------------------"


def smape(act, forc):
    return (
        100 / len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))
    )


def count_chars(string):
    # initialize dictionary to store character counts
    char_counts = {}

    # iterate over each character in the string
    for char in string:
        # check if character is a-z, 0-9 or full stop
        if char.isalpha() and char.islower():
            char_counts[char] = char_counts.get(char, 0) + 1
        elif char.isdigit():
            char_counts[char] = char_counts.get(char, 0) + 1
        elif char == ".":
            char_counts[char] = char_counts.get(char, 0) + 1

    # sort dictionary by key
    sorted_counts = dict(sorted(char_counts.items()))

    # create formatted string
    output = ""
    for char, count in sorted_counts.items():
        output += f"{char}: {count}\n"

    return output


def count_chars_in_file(file_path):
    # initialize dictionary to store character counts for each row
    row_char_counts = {}

    # read in CSV file
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        # skip the header row
        next(reader)
        # iterate over each line in the file
        for row in reader:
            # get the ID value for the row
            row_id = row[0]
            # initialize dictionary to store character counts for this row
            char_counts = {}
            # iterate over each character in the row
            for char in "".join(row[1:]).lower():
                # check if character is a-z, 0-9 or full stop
                if (
                    char.isalpha()
                    and char.islower()
                    and ord(char) >= ord("a")
                    and ord(char) <= ord("z")
                ):
                    char_counts[char] = char_counts.get(char, 0) + 1
                elif char.isdigit():
                    char_counts[char] = char_counts.get(char, 0) + 1
                elif char == ".":
                    char_counts[char] = char_counts.get(char, 0) + 1
            # sort dictionary by key
            sorted_counts = dict(sorted(char_counts.items()))
            # create formatted string for this row
            output = ""
            for char, count in sorted_counts.items():
                output += f"{char}: {count}\n"
            # add formatted string to dictionary with ID as key
            row_char_counts[row_id] = output

    return row_char_counts


def compare_individual_csv_files(file_path_1, file_path_2):

    # count characters in each file
    file1_counts = count_chars_in_file(file_path_1)
    file2_counts = count_chars_in_file(file_path_2)

    # sort row IDs
    sorted_row_ids = sorted(set(file1_counts.keys()) | set(file2_counts.keys()))

    # compare character counts for each row
    smape_values = []
    for row_id in sorted_row_ids:
        # check if row exists in both files
        if row_id not in file1_counts.keys():
            print(f"Row {row_id} does not exist in {file_path_1}")
            continue
        if row_id not in file2_counts.keys():
            print(f"Row {row_id} does not exist in {file_path_2}")
            continue

        # calculate mean absolute error (MAE) and symmetric mean absolute percentage error (SMAPE) for this row
        counts1 = file1_counts[row_id].splitlines()
        counts2 = file2_counts[row_id].splitlines()
        counts1_dict = dict([c.split(":") for c in counts1])
        counts2_dict = dict([c.split(":") for c in counts2])
        all_chars = sorted(set(counts1_dict.keys()) | set(counts2_dict.keys()))
        counts1_list = [int(counts1_dict.get(c, 0)) for c in all_chars]
        counts2_list = [int(counts2_dict.get(c, 0)) for c in all_chars]
        counts1_arr = np.array([int(counts1_dict.get(c, 0)) for c in all_chars])
        counts2_arr = np.array([int(counts2_dict.get(c, 0)) for c in all_chars])
        mae = mean_absolute_error(counts1_list, counts2_list)
        s_mape = smape(counts1_arr, counts2_arr)
        smape_values.append(s_mape)

        # compare character counts for this row
        if file1_counts[row_id] != file2_counts[row_id]:
            print(f"Row {row_id} has different character counts (MAE: {mae}):")
            print(
                f"Row {row_id} has different character counts (SMAPE: {s_mape:.2f}%):"
            )
            print(f"{file_path_1}:".ljust(20), f"{file_path_2}:")
            for line1, line2 in zip_longest(counts1, counts2, fillvalue=""):
                print(line1.ljust(20), line2)

            # # plot histograms
            # chars1 = [c.split(":")[0] for c in counts1]
            # chars2 = [c.split(":")[0] for c in counts2]
            # all_chars = sorted(set(chars1) | set(chars2))
            # counts1_dict = dict([c.split(":") for c in counts1])
            # counts2_dict = dict([c.split(":") for c in counts2])
            # counts1_arr = np.array([int(counts1_dict.get(c, 0)) for c in all_chars])
            # counts2_arr = np.array([int(counts2_dict.get(c, 0)) for c in all_chars])

            # plt.figure(figsize=(10, 5))
            # plt.bar(all_chars, counts1_arr, label=file_path_1, alpha=0.5)
            # plt.bar(all_chars, counts2_arr, label=file_path_2, alpha=0.5)
            # plt.xticks(all_chars)
            # plt.xlabel("Characters")
            # plt.ylabel("Counts")
            # plt.title(f"Character counts for row {row_id}")
            # plt.legend()
            # plt.show()

        else:
            print(f"Row {row_id} has identical character counts in both files ")
    sum_smape = sum(smape_values)
    total_smape_elems = len(smape_values)
    total_smape_avg_score = sum_smape / total_smape_elems
    print(f"Average SMAPE: {total_smape_avg_score:.2f}%")

    # Check if 'SMAPE' column already exists in the existing CSV file.
    with open(file_path_2, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row.
        if 'SMAPE' not in header:
            header.append('SMAPE')
            rows = []
            for row in reader:
                if row[0] in file2_counts.keys():
                    s_mape = smape_values[sorted_row_ids.index(row[0])]
                    row.append(str(s_mape))
                else:
                    row.append('')
                rows.append(row)
        else:
            # 'SMAPE' column already exists, no need to update.
            return

    with open(file_path_2, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


def print_individual_count(results):
    for row_id, char_counts in results.items():
        print(f"Row {row_id} character counts:")
        print(char_counts)
        print()


"------------------------ADAPTIVE GAMMA CORRECTION-----------------------"

def adaptive_gamma_correction(image, block_size=64, gamma=1.0):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i:i+block_size, j:j+block_size]
            blocks.append(block)

    # Calculate the gamma-corrected image block by block
    gamma_corrected_blocks = []
    for block in blocks:
        # Calculate the histogram of the block
        hist, _ = np.histogram(block, bins=256, range=(0, 256))

        # Calculate the cumulative distribution function of the block
        cdf = hist.cumsum()

        # Normalize the cumulative distribution function
        cdf_normalized = cdf / cdf[-1]

        # Calculate the gamma-corrected lookup table for the block
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Apply the gamma-corrected lookup table to the block, weighted by the probability of each intensity value
        weighted_table = cv.LUT(table, cdf_normalized * 255.0)
        gamma_corrected_block = cv.LUT(block, table)

        # Add the gamma-corrected block to the list of gamma-corrected blocks
        gamma_corrected_blocks.append(gamma_corrected_block)

    # Assemble the gamma-corrected image by merging the gamma-corrected blocks
    gamma_corrected_image = np.zeros_like(gray)
    block_index = 0
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            gamma_corrected_block = gamma_corrected_blocks[block_index]
            gamma_corrected_image[i:i+block_size, j:j+block_size] = gamma_corrected_block
            block_index += 1

    return gamma_corrected_image


def adaptive_gamma_correction_with_otsu(image, block_size=16, gamma=1.0):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i : i + block_size, j : j + block_size]
            blocks.append(block)

    # Apply Otsu's thresholding to each block and calculate the gamma-corrected image
    gamma_corrected_blocks = []
    for block in blocks:
        # Apply Otsu's thresholding to the block
        _, thresholded_block = cv.threshold(
            block, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        # Calculate the histogram of the thresholded block
        hist, _ = np.histogram(thresholded_block, bins=256, range=(0, 256))

        # Calculate the cumulative distribution function of the thresholded block
        cdf = hist.cumsum()

        # Normalize the cumulative distribution function
        cdf_normalized = cdf / cdf[-1]

        # Calculate the gamma-corrected lookup table for the thresholded block
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # Apply the gamma-corrected lookup table to the thresholded block
        gamma_corrected_block = cv.LUT(thresholded_block, table)

        # Add the gamma-corrected block to the list of gamma-corrected blocks
        gamma_corrected_blocks.append(gamma_corrected_block)

    # Assemble the gamma-corrected image by merging the gamma-corrected blocks
    gamma_corrected_image = np.zeros_like(gray)
    block_index = 0
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            gamma_corrected_block = gamma_corrected_blocks[block_index]
            gamma_corrected_image[
                i : i + block_size, j : j + block_size
            ] = gamma_corrected_block
            block_index += 1

    return gamma_corrected_image

"Convert String to Dictionary"
def string_to_dict(string):
    # initialize dictionary to store character counts
    char_counts = {}

    # iterate over each character in the string
    for char in string:
        # check if character is a-z, 0-9 or full stop
        if char.isalpha() and char.islower():
            char_counts[char] = char_counts.get(char, 0) + 1
        elif char.isdigit():
            char_counts[char] = char_counts.get(char, 0) + 1
        elif char == ".":
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # sort dictionary by key
    sorted_counts = dict(sorted(char_counts.items()))

    return sorted_counts