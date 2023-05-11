import preprocessing as prep
import cv2 as cv
import os

# filename = "./sample_images/nf131.jpg"
# Load the image
image = cv.imread("./sample_images/nf131.jpg")
cv.imshow("Original", image)

deskew = prep.deskew_image(image)
cv.imshow("Original Deskew", deskew)

# Resize the image using the first method
resized_image1 = prep.resize_image(image)
cv.imshow("Resized 1", resized_image1)


# Resize the image using the second method
resized_image2 = prep.resize_image2(image)
cv.imshow("Resized 2", resized_image2)


cv.waitKey(0)
cv.destroyAllWindows()