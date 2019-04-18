from PIL import Image
import sys

import pyocr
import pyocr.builders

import cv2 as cv
import os



tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.

def getText(image):
    text = tool.image_to_string(
        Image.open(image),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    return text

def getTextFeed(path):

    if('frames' not in os.listdir()):
        os.mkdir('frames')

    cap = cv.VideoCapture(path)

    while(1):
        
        ret, frame = cap.read()

        frame = frame[0:60, 0:160]
        cv.imshow('video', frame)

        # imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('grayscale', imgray)
        # ret, thresh = cv.threshold(imgray, 150, 255, 0)
        # cv.imshow('threshold', thresh)
        # enlarged = cv.resize(thresh, (0,0), fx=5, fy=5) 
        # cv.imshow('enlarged', enlarged)
        
        
        # # dilation = cv.dilate(enlarged, kernel, iterations = 3)
        # # cv.imshow('Dilation', dilation)

        # laplacian = enlarged - cv.Laplacian(enlarged, cv.CV_64F)
        # cv.imshow('laplacian', laplacian)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        # erosion = cv.erode(laplacian, kernel, iterations = 1)
        # cv.imshow('Erosion', erosion)

        cv.imwrite('./frames/frame.png', frame)

        text = getText('./frames/frame.png').replace(' ', '')
        print(text)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
        # time.sleep(0.5)
    
    cap.release()
    cv.destroyAllWindows()

path = './traffic-time.mp4'
getTextFeed(path)