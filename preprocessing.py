import cv2 as cv
import numpy as np


# function to convert image to grayscale
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# function to remove noise from image
def remove_noise(image):
    return cv.medianBlur(image, 5)


# function to find edges, convert to binary image
def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


# function to deskew the image
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
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
