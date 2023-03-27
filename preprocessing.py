import re
import cv2 as cv
import numpy as np
import pytesseract
from pytesseract import Output
import csv
import os


# function to convert image to grayscale
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def thresholding(image):
    # Apply binary thresholding using Otsu's method to the input image and return the thresholded image
    # The parameters 0 and 255 set the pixel intensity values for the black and white pixels, respectively
    # cv.THRESH_BINARY specifies that a binary thresholding is used
    # cv.THRESH_OTSU indicates that Otsu's method is used to determine the threshold value automatically
    thresholded_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # Return the thresholded image
    return thresholded_image


# function to deskew the image
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(
            image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE
        )
        return rotated


def match_template(image, template):
    return cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)


# function to resize images
def image_resize(image):
    print("Original Size: ", image.shape)
    image = cv.resize(image, (400, 400))
    print("New Size: ", image.shape)
    return image


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image


def boundingboxes(image):
    # Define custom Tesseract configuration for recognizing text
    custom_config = r'--oem 3 --psm 11'

    # Use Tesseract to get text data from the input image
    details = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config, lang='eng')

    # Print keys of the dictionary returned by Tesseract
    print(details.keys())

    # Get total number of text boxes detected by Tesseract
    total_boxes = len(details['text'])

    # Loop through each text box and draw a bounding box around it if the confidence level is high enough
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > 30:  # set minimum confidence level to 30
            # Extract the bounding box coordinates from the dictionary returned by Tesseract
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            # Draw a rectangle around the text box on the input image using OpenCV
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes drawn
    cv.imshow('captured data', image)
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
    last_word = ''

    # Iterate through each word in the text of the details dictionary
    for word in details['text']:

        # If the current word is not an empty string, append it to the current line of text
        if word != '':
            word_list.append(word)

            # Update the last word seen to be the current word
            last_word = word

        # If the current word is an empty string and the last word seen was not empty,
        # or if the current word is the last word in the text, then the current line of text is complete
        # and can be added to the parsed text list
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)

            # Reset the word list to empty for the next line of text
            word_list = []

    # Print the parsed text list
    print(parse_text)


"-------------------------------------------------------------------"

def preprocess_image(image):
    """
    Preprocesses an input image for OCR.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The preprocessed image as a binary NumPy array.
    """

    # Resize the image to a fixed size of 400x400 pixels.
    image = cv.resize(image, (400, 400))

    # Convert the image to grayscale.
    gray = cv.cvtColor(np.array(image), cv.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to the grayscale image to obtain a binary image.
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

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
    calorie_pattern = re.compile(r'Calories[:\s]*(\d+)')
    fat_pattern = re.compile(r'Fat[:\s]*(\d+)')
    carb_pattern = re.compile(r'Carbohydrate[:\s]*(\d+)')
    protein_pattern = re.compile(r'Protein[:\s]*(\d+)')

    # Search for the first occurrence of each pattern in the OCR output text.
    calories = calorie_pattern.search(text)
    fat = fat_pattern.search(text)
    carbs = carb_pattern.search(text)
    protein = protein_pattern.search(text)

    # Extract the numerical values that match each pattern, or assign 'N/A' if no match is found.
    if calories:
        calories = calories.group(1)
    else:
        calories = 'N/A'

    if fat:
        fat = fat.group(1)
    else:
        fat = 'N/A'

    if carbs:
        carbs = carbs.group(1)
    else:
        carbs = 'N/A'

    if protein:
        protein = protein.group(1)
    else:
        protein = 'N/A'

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
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row to the CSV file.
        writer.writerow(['Image File', 'Character Count', 'Calories', 'Fat', 'Carbs', 'Protein'])

        # Process each file in the input folder.
        for filename in os.listdir(input_folder):

            # Only process files with the '.jpg' or '.png' extension.
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load the image and preprocess it.
                image_file = os.path.join(input_folder, filename)
                image = cv.imread(image_file)
                processed_image = preprocess_image(image)

                # Perform OCR on the preprocessed image and extract the nutritional information.
                ocr_output = pytesseract.image_to_string(processed_image)
                calories, fat, carbs, protein = extract_nutrition_info(ocr_output)
                character_count = count_characters(ocr_output)

                # Write the extracted nutritional information and character count to the CSV file.
                writer.writerow([filename, character_count, calories, fat, carbs, protein])

