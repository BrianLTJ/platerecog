import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imgs/04_1.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

blur = cv2.GaussianBlur(thresh,(5,5),0)


cv2.imshow("thresh",thresh)

im2, contours, hierarchy = cv2.findContours(blur,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)


cv2.imshow("img",img)
cv2.imshow("gray",gray)
cv2.imshow("blur",blur)
cv2.waitKey(15*1000)
