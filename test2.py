import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import matplotlib.pyplot as plt


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image,kernel,iterations = 1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image,kernel,iterations = 1)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return  cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90+angle)
    else:
        angle = -angle
        (h,w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)



img = cv2.imread('sample_nutrilabel.jpg')
print('image w: ',img.shape[1])
print('image h: ',img.shape[0])
cv2.imshow('img', img)
cv2.waitKey(0)
img = cv2.resize(img,(400,400))
print('image w: ',img.shape[1])
print('image h: ',img.shape[0])
cv2.imshow('img', img)
cv2.waitKey(0)

image = cv2.imread('sample_nutrilabel_300.jpg')
print(image.shape)
image_res = cv2.resize(image,(400,400))
print(image_res.shape)
image2 = cv2.imread('sample_nutrilabel.jpg')
print(image2.shape)
gray2 = get_grayscale(image2)


# h,w,c = image.shape
# boxes = pytesseract.image_to_boxes(image)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0,255,0),2)
cv2.imshow('img', image)
cv2.waitKey(0)

gray = get_grayscale(image_res)

gray_data = pytesseract.image_to_data(gray, output_type=Output.DICT)
print(gray_data.keys())

cv2.imshow('img', gray)
cv2.waitKey(0)

#getting boxes arround text
gray_boxes = len(gray_data['text'])
for i in range(gray_boxes):
    if int(gray_data['conf'][i])>8:
        (x,y,w,h)=(gray_data['left'][i],gray_data['top'][i], gray_data['width'][i],gray_data['height'][i])
        gray = cv2.rectangle(gray, (x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('img',gray)
cv2.waitKey(0)


#must be in grayscale to work
thresh = thresholding(gray2)
# cv2.imshow('img', thresh)
# cv2.waitKey(0)

thresh_data = pytesseract.image_to_data(thresh,output_type=Output.DICT)
print(thresh_data.keys())
#text template matching
pattern = 'Energy'
thresh_boxes = len(thresh_data['text'])
for i in range(thresh_boxes):
    if int(thresh_data['conf'][i]>60):
        if re.match(pattern, thresh_data['text'][i]):
            (x, y, w, h) = (thresh_data['left'][i], thresh_data['top'][i], thresh_data['width'][i], thresh_data['height'][i])
            thresh = cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('img', thresh)
cv2.waitKey(0)

opening = opening(gray)
# cv2.imshow('img', opening)
# cv2.waitKey(0)

canny = canny(gray)
# cv2.imshow('img', canny)
# cv2.waitKey(0)

#Detect orientation
rotated = cv2.imread('sample_nutrilabel_rotated.jpg')
osd = pytesseract.image_to_osd(rotated)
angle = re.search('(?<=Rotate: )\d+', osd).group(0)
# script = re.search('(?<=Script: )\d+', osd).group(0)
print("angle: ", angle)
# print("script: ", script)

## testing git
## testing 2
#finall test