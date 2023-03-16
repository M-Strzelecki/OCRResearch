import preprocessing as prep
import pytesseract
import cv2 as cv
import os

# importing image
image = cv.imread("./sample_images/nf157.jpg")
file_name = os.path.basename("./sample_images/nf157.jpg")
image_name = os.path.splitext(file_name)[0]

# print image and image dimensions
cv.imshow(image_name, image)
cv.waitKey(0)

# resizing image and printing dimensions
prep.image_resize(image)

# print image after resizing
cv.imshow("image", image)
cv.waitKey(0)

beforegray = cv.bitwise_not(image)
cv.imshow("before gray", beforegray)
cv.waitKey(0)

# converting image to grayscale
gray = prep.get_grayscale(image)
cv.imshow(image_name, gray)
cv.waitKey(0)

aftergray = cv.bitwise_not(gray)
cv.imshow("after gray", aftergray)
cv.waitKey(0)

# converting image to binary
binary = prep.thresholding(gray)
cv.imshow(image_name, binary)
cv.waitKey(0)

binarywithinvert = prep.thresholding(aftergray)
cv.imshow("binary after invert", binarywithinvert)
cv.waitKey(0)

invertafterbinary = cv.bitwise_not(binary)
cv.imshow("invert after binary", invertafterbinary)
cv.waitKey(0)
