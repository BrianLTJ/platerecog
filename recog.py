import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imgs/04_1.png")

# gaussina blur
blur = cv2.GaussianBlur(img,)

# gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobel

# otsu threshold


thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,2)

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

possiblePlateArea = []

for cont in contours:
    (x,y,w,h) = cv2.boundingRect(cont)
    ar = w / float(h)
    if w >=30 and h>=30 and ar >=1.3 and ar <=3.5:
        possiblePlateArea.append(cont)

# cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.drawContours(img, possiblePlateArea, -1, (0,255,0), 3)
# cv2.imshow("thresh",thresh)
cv2.imshow("img",img)
# cv2.imshow("gray",gray)
# cv2.imshow("blur",blur)
cv2.waitKey(5*1000)
