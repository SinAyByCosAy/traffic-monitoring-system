import numpy as np
import cv2
import time
import pytesseract

imgpath = './sample-text.png'

def getTimeFeed(path):

    cap = cv.VideoCapture(path)

    while(1):
        
        ret, frame = cap.read()
        
        # fgmask = fgbg.apply(frame)
        # cv.imshow('BG-Subtraction', fgmask)
        
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        # # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        # # kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

        # dilation = cv.dilate(fgmask, kernel, iterations = 5)
        # cv.imshow('Dilation', dilation)

        # erosion = cv.erode(dilation, kernel, iterations = 5)
        # cv.imshow('Erosion', erosion)

        
        # # opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
        # # cv.imshow('Opening', opening)


        # # closing = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        # # cv.imshow('Closing', closing)

        # # ret, thresh = cv.threshold(fgmask, 127, 255, 0)
        # im2, contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        # # cv.drawContours(fgmask, contours, -1, (0,255,0), 1)
        # # cv.imshow('contours', im2)

        # cv.line(frame, (0,275), (700,275), (255,0,0), 1)

        # for contour in contours:
        #     x, y, w, h = cv.boundingRect(contour)
        #     if w<15 or h<15:
        #         continue
        #     # if w/h>1.5 or h/w>1.5:
        #     #     continue
        #     cx = x + int(w/2)
        #     cy = y + int(h/2)
        #     # print(x, y, w, h, cx, cy)
        #     if (abs(cy-275)<=1):
        #         print(cx, cy)
        #         if(cx<320):
        #             leftcount += 1
        #         else:
        #             rightcount += 1
        #     print(leftcount, rightcount)
        #     text_left = 'Left: {}'.format(leftcount)
        #     text_right = 'Right: {}'.format(rightcount)
        #     font = cv.FONT_HERSHEY_SIMPLEX
        #     cv.putText(frame, text_left , (10,50), font, 1 ,(255,0,0), 1, cv.LINE_AA)
        #     cv.putText(frame, text_right , (200,50), font, 1 ,(255,0,0), 1, cv.LINE_AA)
        #     cv.circle(frame, (cx,cy), 1, (0,0,255), -1)
        #     cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)


        cv.imshow('video', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
        # time.sleep(0.5)
    
    cap.release()
    cv.destroyAllWindows()

def getText(path):
    imPath = path
     
    config = ('-l eng --oem 1 --psm 3')

    # Read image from disk
    im = cv2.imread(imPath, cv2.IMREAD_COLOR)

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(im, config=config)

    # Print recognized text
    print(text)

getText(imgpath)