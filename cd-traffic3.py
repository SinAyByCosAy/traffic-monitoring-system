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

# tools = pyocr.get_available_tools()
# if len(tools) == 0:
#     print("No OCR tool found")
#     sys.exit(1)
# # The tools are returned in the recommended order of usage
# tool = tools[0]
# print("Will use tool '%s'" % (tool.get_name()))
# # Ex: Will use tool 'libtesseract'

# langs = tool.get_available_languages()
# print("Available languages: %s" % ", ".join(langs))
# lang = langs[0]
# print("Will use lang '%s'" % (lang))

# def getDistance(p1, p2):
#     return round(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5, 3)

# vehicle = {
#     'frames': None,
#     'timein': None,
#     'pointin': None,
#     'time': None,
#     'point': None,
#     'speed': None,
#     'type': None,
#     'area': None, 
#     'brect': None
# }

# vehicle_list = []
# vehicle_count = {
#     'left': 0,
#     'right': 0
# }

# def addVehicle(newveh, vehicle_list):  
#     if len(vehicle_list)==0:
#         vehicle_list.append(newveh)
#         return vehicle_list
#     for vehicle in vehicle_list:
#         if getDistance(newveh['point'], vehicle['point'])<10:
#             vehicle['frames'] += 1
#             vehicle['time'] = newveh['time']-vehicle['timein']
#             vehicle['distance'] = getDistance(newveh['point'], vehicle['point'])
#             vehicle['point'] = newveh['point']
#             if vehicle['time']:
#                 vehicle['speed'] = round((vehicle['distance']/vehicle['time'])*1.0, 3)
#             else:
#                 vehicle['speed'] = 0
#             vehicle['brect'] = newveh['brect']
#             vehicle['area'] = newveh['brect'][2]*newveh['brect'][3]
#             vehicle['type'] = 'heavy' if vehicle['area'] > 10000 else 'light'
#             return vehicle_list
#     vehicle_list.append(newveh)
#     return vehicle_list


# def updateVehicleList(vehicle_list, vehicle_count):
#     new_list = []
#     for vehicle in vehicle_list:
#         if vehicle['point'][1]<230 and vehicle['distance']>80:
#             vehicle_count['left'] += 1
#         elif vehicle['point'][1]>330 and vehicle['distance']>80:
#             vehicle_count['right'] += 1
#         elif vehicle['frames'] >= (vehicle['time']-vehicle['timein'])*25:
#             new_list.append(vehicle)
#     vehicle_list = new_list
#     return [vehicle_list, vehicle_count]

# def getTime(frame):
#     frame = frame[0:60, 0:300]
#     image = './frames/frame.png'
#     cv.imwrite(image, frame)
#     text = tool.image_to_string(
#         Image.open(image),
#         lang=lang,
#         builder=pyocr.builders.TextBuilder()
#     )
#     text = text.replace(' ', '')
#     print('OCR text', text)
#     h, m, s = list(map(int, text.split(':')))
#     return (h,m,s)

# def draw(frame, vehicle_list, vehicle_count):
#     cv.line(frame, (0,275), (700,275), (255,0,0), 1)
#     cv.line(frame, (0,230), (700,230), (255,0,0), 1)
#     cv.line(frame, (0,330), (700,330), (255,0,0), 1)
#     for vehicle in vehicle_list:
#         color = (0,255,0)
#         if 'type' in vehicle and vehicle['type']=='heavy':
#             color = (0,0,255)
#         x, y, w, h = vehicle['brect']
#         cx, cy = vehicle['point']
#         cv.rectangle(frame, (x,y), (x+w,y+h), color, 1)
#         # cv.circle(frame, (cx,cy), 2, color, -1)
#         cv.putText(frame, str(vehicle['distance']) , (cx,cy), cv.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255), 1, cv.LINE_AA)

#     return frame

# def processVideo(path, vehicle_list, vehicle_count):
#     cap = cv.VideoCapture(path)
#     fgbg = cv.createBackgroundSubtractorMOG2()
#     frame_count = 0
#     mat = []
#     while(1):
#         ret, frame = cap.read()
#         frame_count += 1
#         hr, mn, sc = getTime(frame)
#         time = hr*3600 + mn*60 + sc
#         # print('Time of frame - {:02}:{:02}:{:02}'.format(hr, mn, sc))
#         # print(vehicle_count)
#         fgmask = fgbg.apply(frame)
#         cv.imshow('BG-Subtraction', fgmask)
#         ret, thresh = cv.threshold(fgmask, 254, 255, 0)
#         cv.imshow('threshold', thresh)
#         kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
#         closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
#         cv.imshow('Closing', closing)
#         opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
#         cv.imshow('Opening', opening)
#         im2, contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
#         for contour in contours:    
#             x, y, w, h = cv.boundingRect(contour)
#             if w<25 or h<25:
#                 continue
#             # # if w/h>1.5 or h/w>1.5:
#             #     continue
#             cx = x + int(w/2)
#             cy = y + int(h/2)
#             # print(row)
#             # mat.append(row)
#             if cy<175:
#                 continue
#             newveh = {
#                 'frames': 0,
#                 'timein': time,
#                 'pointin': (cx, cy),
#                 'time': time,
#                 'point': (cx, cy),
#                 'distance': 0,
#                 'brect': (x, y, w, h)
#             }
            
#             vehicle_list = addVehicle(newveh, vehicle_list)
#         vehicle_list, vehicle_count = updateVehicleList(vehicle_list, vehicle_count)
#         frame = draw(frame, vehicle_list, vehicle_count)
#         cv.imshow('frame', frame)
#         k = cv.waitKey(30) & 0xff
#         if k == 27:
#             break

#     # df = pd.DataFrame(mat, columns=['Time', 'Frame', 'x', 'y', 'w', 'h', 'cx', 'cy'])
#     # df.to_csv('vehicle_data.csv')
#     cap.release()
#     cv.destroyAllWindows()





def mog1(path):
    cap = cv.VideoCapture(path)
    fgbg = cv.createBackgroundSubtractorMOG2()

    leftcount, rightcount = 0, 0

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
        
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
        # kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        
        # dilation = cv.dilate(fgmask, kernel, iterations = 5)
        # cv.imshow('Dilation', dilation)

        # erosion = cv.erode(dilation, kernel, iterations = 5)
        # cv.imshow('Erosion', erosion)

        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        cv.imshow('Closing', closing)

        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        cv.imshow('Opening', opening)

        #im2, contours, heirarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        contours,hierachy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        # cv.drawContours(fgmask, contours, -1, (0,255,0), 1)
        # cv.imshow('contours', fgmask)

        cv.line(frame, (0,275), (700,275), (255,0,0), 1)
        cv.line(frame, (0,230), (700,230), (255,0,0), 1)
        cv.line(frame, (0,330), (700,330), (255,0,0), 1)

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
            print(leftcount, rightcount)
            text_left = 'Left: {}'.format(leftcount)
            text_right = 'Right: {}'.format(rightcount)
            font = cv.FONT_HERSHEY_SIMPLEX
            reccolor = (0,255,0)
            if (w*h)>=6400:
                reccolor = (0,0,255)
            cv.putText(frame, text_left , (230,30), font, 1 ,(255,0,0), 1, cv.LINE_AA)
            cv.putText(frame, text_right , (400,30), font, 1 ,(255,0,0), 1, cv.LINE_AA)
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