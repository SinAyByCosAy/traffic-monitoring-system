import numpy as np
import cv2

impath = './images/im5.jpeg'
im = cv2.imread(impath)
cv2.imshow('Original image', im)
# cv2.resizeWindow('Original image', 1280, 720)
cv2.waitKey()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', imgray)
cv2.waitKey()

ret, thresh = cv2.threshold(imgray, 16, 255, cv2.THRESH_BINARY)
print(ret)
cv2.imshow('Binary thresh', thresh)
cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

erosion = cv2.erode(thresh, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey()

dilation = cv2.dilate(thresh, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv2.waitKey()

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey()

closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.waitKey()

im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('Contour', im2)
print(contours)
print(hierarchy)
cv2.waitKey()

img = im
# cv2.drawContours(img, contours, -1, (0,255,0), 3)
# cv2.imshow('Contours drawing', img)
# cv2.waitKey()

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + int(w/2)
    cy = y + int(h/2)
    print(x, y, w, h, cx, cy)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)

cv2.imshow('Rects', img)
cv2.waitKey()
