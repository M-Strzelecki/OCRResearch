import test2 as ocrfun
import pytesseract
import cv2

image = cv2.imread('./sample_images/nf157.jpg')
cv2.imshow('image',image)
cv2.waitKey(0)
ocrfun.image_resize(image)

cv2.imshow('image',image)
cv2.waitKey(0)