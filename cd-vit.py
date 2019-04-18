# Gaussian mixture based bg/fg segmentation algorithm

import numpy as np
import cv2 as cv
from threading import Thread
import time

def mog1(path):
    cap = cv.VideoCapture(path)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

    leftcount, rightcount = 0, 0
    
    while(1):
        
        ret, frame = cap.read()
        
        fgmask = fgbg.apply(frame)
        cv.imshow('BG-Subtraction', fgmask)
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        # kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

        dilation = cv.dilate(fgmask, kernel, iterations = 10)
        cv.imshow('Dilation', dilation)

        erosion = cv.erode(dilation, kernel, iterations = 4)
        cv.imshow('Erosion', erosion)

        
        # opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
        # cv.imshow('Opening', opening)


        # closing = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        # cv.imshow('Closing', closing)

        # ret, thresh = cv.threshold(fgmask, 127, 255, 0)
        im2, contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        # cv.drawContours(fgmask, contours, -1, (0,255,0), 1)
        # cv.imshow('contours', im2)

        cv.line(frame, (450,230), (700,230), (255,0,0), 1)
        cv.line(frame, (280,260), (500,260), (255,0,0), 1)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w<15 or h<15:
                continue
            # if w/h>1.5 or h/w>1.5:
            #     continue
            cx = x + int(w/2)
            cy = y + int(h/2)
            # print(x, y, w, h, cx, cy)
            if (abs(cy-230)<=1 and cx>400):
                print(cx, cy)
                rightcount += 1
            elif(abs(cy-260)<=1 and (cx>280 and cx<500)):
                print(cx,cy)
                leftcount += 1
            print(leftcount, rightcount)
            text_left = 'Left: {}'.format(leftcount)
            text_right = 'Right: {}'.format(rightcount)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, text_left , (10,50), font, 1 ,(255,0,0), 1, cv.LINE_AA)
            cv.putText(frame, text_right , (200,50), font, 1 ,(255,0,0), 1, cv.LINE_AA)
            cv.circle(frame, (cx,cy), 1, (0,0,255), -1)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)


        cv.imshow('Rects', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
        # time.sleep(0.5)
    
    cap.release()
    cv.destroyAllWindows()

def mog2(path):
    cap = cv.VideoCapture(path)
    fgbg = cv.createBackgroundSubtractorMOG2()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('mog2',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

def bayesian(path):
    cap = cv.VideoCapture(path)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        cv.imshow('bayesian',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

def compare(path):
    mog1(path)
    mog2(path)
    bayesian(path)

file_path = './s3v2.mp4'
mog1(file_path)