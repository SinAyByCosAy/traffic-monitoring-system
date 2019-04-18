# Gaussian mixture based bg/fg segmentation algorithm

import numpy as np
import cv2 as cv
from threading import Thread
# import time
from PIL import Image
import sys
import os
import pyocr
import pyocr.builders
import pandas


def mog1(path):
    cap = cv.VideoCapture(path)
    fgbg = cv.createBackgroundSubtractorMOG2()

    leftcount, rightcount = 0, 0
    count_car = 0;
    count_heavy = 0;
    frame_count = 0
    
    while(1):
        
        ret, frame = cap.read()
        frame_count += 1
        
        fgmask = fgbg.apply(frame)
        cv.imshow('BG-Subtraction', fgmask)

        ret, thresh = cv.threshold(fgmask, 254, 255, 0)
        cv.imshow('threshold', thresh)

        thresh = cv.GaussianBlur(thresh, (5, 5), 0)
        cv.imshow('blur', thresh)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        cv.imshow('Closing', closing)

        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        cv.imshow('Opening', opening)

        contours,hierachy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        

        # cv.line(frame, (0,275), (700,275), (255,0,0), 1)
        # cv.line(frame, (0,230), (700,230), (255,0,0), 1)
        # cv.line(frame, (0,330), (700,330), (255,0,0), 1)

        for contour in contours:    
            x, y, w, h = cv.boundingRect(contour)
            if w<25 or h<25:
                continue
            # if w/h>1.5 or h/w>1.5:
            #     continue
            cx = x + int(w/2)
            cy = y + int(h/2)
            if cy<175:
                continue

            print(x, y, w, h, cx, cy)
            if (abs(cy-275)<=1):
                print(cx, cy)
                if(cx<320):
                    leftcount += 1
                else:
                    rightcount += 1
                if (w*h)>=6400:
                    reccolor = (0,0,255)
                    count_heavy+=1;
                else:
                    count_car+=1;
            print(leftcount, rightcount)
            text_left = 'Left: {}'.format(leftcount)
            text_right = 'Right: {}'.format(rightcount)
            font = cv.FONT_HERSHEY_SIMPLEX
            reccolor = (0,255,0)

            

            text_car = 'Light Vehicles:{}'.format(count_car)
            text_heavy = 'Heavy Vehicles:{}'.format(count_heavy)
            cv.putText(frame, text_left , (230,30), font, 1 ,(208,41,45), 1, cv.LINE_AA)
            cv.putText(frame, text_right , (400,30), font, 1 ,(208,41,45), 1, cv.LINE_AA)

            cv.putText(frame, text_car, (40,300), font, 1 ,(138,241,174), 1, cv.LINE_AA)
            cv.putText(frame, text_heavy , (40,330), font, 1 ,(138,241,174), 1, cv.LINE_AA)

            cv.circle(frame, (cx,cy), 1, (0,0,255), -1)
            cv.rectangle(frame, (x,y), (x+w,y+h), reccolor, 1)

            
        
        

        cv.imshow('Rects', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
        #time.sleep(0.5)
    
    cap.release()
    cv.destroyAllWindows()

file_path = './s3v2.mp4'
mog1(file_path)