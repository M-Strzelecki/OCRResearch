import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):
    # apply histogram equalization to improve contrast
    image = cv2.equalizeHist(image)

    # apply gamma correction to adjust gamma levels
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def automatic_gamma_correction(image, percentile=0.1):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    # calculate the cumulative distribution function
    cdf = hist_norm.cumsum()

    # determine the percentile of pixels to use for gamma correction
    percentile_value = np.percentile(gray, percentile * 100)

    # calculate the gamma value
    gamma = np.log10(0.5) / np.log10(percentile_value / 255.0)

    # apply gamma correction to adjust gamma levels
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def adaptive_gamma_correction(image, block_size=32, gamma=1.0):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i : i + block_size, j : j + block_size]
            blocks.append(block)

    # Calculate the gamma-corrected image by applying adaptive gamma correction to each block
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
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # Apply the gamma-corrected lookup table to the block
        gamma_corrected_block = cv2.LUT(block, table)

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


def adaptive_gamma_correction_with_otsu(image, block_size=32, gamma=1.0):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        _, thresholded_block = cv2.threshold(
            block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
        gamma_corrected_block = cv2.LUT(thresholded_block, table)

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


def adaptive_gamma_correction_with_otsu_clarity(image, block_size=64, gamma=1.5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Divide the image into blocks
    blocks = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i : i + block_size, j : j + block_size]
            blocks.append(block)

    # Apply Otsu's thresholding and morphological opening to each block, and calculate the gamma-corrected image
    gamma_corrected_blocks = []
    for block in blocks:
        # Apply Otsu's thresholding to the block
        _, thresholded_block = cv2.threshold(
            block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Apply a morphological opening operation to the thresholded block
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened_block = cv2.morphologyEx(thresholded_block, cv2.MORPH_OPEN, kernel)

        # Calculate the histogram of the opened block
        hist, _ = np.histogram(opened_block, bins=256, range=(0, 256))

        # Calculate the cumulative distribution function of the opened block
        cdf = hist.cumsum()

        # Normalize the cumulative distribution function
        cdf_normalized = cdf / cdf[-1]

        # Calculate the gamma-corrected lookup table for the opened block
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # Apply the gamma-corrected lookup table to the opened block
        gamma_corrected_block = cv2.LUT(opened_block, table)

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


# Load the image
img = cv2.imread("./sample_images/nf131.jpg")

# # Calculate the mean brightness of the image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# mean_brightness = np.mean(gray)
# print(mean_brightness)
# # Adjust the gamma level based on the brightness
# gamma = 1.0
# if mean_brightness < 100:
#     gamma = 1.5
# elif mean_brightness < 150:
#     gamma = 1.2
# elif mean_brightness < 200:
#     gamma = 0.8
# else:
#     gamma = 0.5
#
# # Apply the gamma correction
# inv_gamma = 1.0 / gamma
# table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
# img_gamma_corrected = cv2.LUT(img, table)
#
# # Display the original and gamma-corrected images
# cv2.imshow('Original Image', img)
# cv2.imshow('Gamma Corrected Image', img_gamma_corrected)
#
#
#
# gray_image = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2GRAY)
# cv2.imshow("New Gray Image", gray_image)

adapt_otsu = adaptive_gamma_correction_with_otsu(img)
cv2.imshow("New Adapt with Otsu", adapt_otsu)

adapt_otsu_clarity = adaptive_gamma_correction_with_otsu_clarity(img)
cv2.imshow("New Adapt with Otsu+Clarity", adapt_otsu_clarity)

auto_gama = automatic_gamma_correction(img)
cv2.imshow("Auto Gamma Corrected Image", auto_gama)
auto_gama_gray = cv2.cvtColor(auto_gama, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.threshold(auto_gama_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("Auto Threshed Image", thresh2)

adap_gama = adaptive_gamma_correction(img)
# adap_gama_gray = cv2.cvtColor(adap_gama, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.threshold(adap_gama, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("Adapt Threshed Image", thresh2)

cv2.waitKey(0)
cv2.destroyAllWindows()
