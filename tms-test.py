import numpy as np
import cv2 as cv
from PIL import Image
import sys
import os
import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))

def getDistance(p1, p2):
    return round(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5, 3)

vehicle = {
    'frames': None,
    'timein': None,
    'pointin': None,
    'time': None,
    'point': None,
    'speed': None,
    'type': None,
    'area': None, 
    'brect': None
}

vehicle_list = []
vehicle_count = {
    'left': 0,
    'right': 0
}


def addVehicle(newveh, vehicle_list, frameid):  
    
    if len(vehicle_list)==0:
        vehicle_list.append(newveh)
        return vehicle_list
    
    for vehicle in vehicle_list:
        d = getDistance(newveh['point'], vehicle['point']) 
        if d>1 and d<100:
            vehicle['frameid'] = newveh['frameid']
            vehicle['frames_since'] = 0
            vehicle['time'] = newveh['time']-vehicle['timein']
            vehicle['point'] = newveh['point']
            vehicle['distance'] = getDistance(vehicle['point'], vehicle['pointin'])
            # vehicle['point'] = newveh['point']
            if vehicle['time']:
                vehicle['speed'] = round((vehicle['distance']/vehicle['time'])*1.0, 3)
            else:
                vehicle['speed'] = 0
            vehicle['brect'] = newveh['brect']
            vehicle['area'] = newveh['brect'][2]*newveh['brect'][3]
            vehicle['type'] = 'heavy' if vehicle['area'] > 10000 else 'light'
            vehicle['counted'] = newveh['counted']
            return vehicle_list
        vehicle['frame_since'] = frameid - vehicle['frameid'] 
    
    vehicle_list.append(newveh)
    return vehicle_list


def updateVehicleList(vehicle_list, vehicle_count):
    new_list = []
    
    for vehicle in vehicle_list:
        if vehicle['point'][0]<325 and vehicle['point'][1]<230:
            if vehicle['counted']==False:
                vehicle_count['left'] += 1
                vehicle['counted']=True
            new_list.append(vehicle)
            print('Dropping vehicle on left lane.')
            print(vehicle)
            continue
        elif vehicle['point'][0]>325 and vehicle['point'][1]>330:
            if vehicle['counted']==False:
                vehicle_count['right'] += 1
                vehicle['counted']=True
            new_list.append(vehicle)
            print('Dropping vehicle on right lane.')
            print(vehicle)
            continue
        if vehicle['frames_since'] > 2:
            print('Dropping vehicle for vanishing.')
            print(vehicle)
            continue
        new_list.append(vehicle)
    
    vehicle_list = new_list
    return [vehicle_list, vehicle_count]


def getTime(frame):
    frame = frame[0:60, 0:300]
    image = './frames/frame.png'
    cv.imwrite(image, frame)
    text = tool.image_to_string(
        Image.open(image),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    text = text.replace(' ', '')
    print('OCR text', text)
    h, m, s = list(map(int, text.split(':')))
    return (h,m,s)


def draw(frame, vehicle_list, vehicle_count):
    cv.line(frame, (0,275), (700,275), (255,0,0), 1)
    cv.line(frame, (0,230), (700,230), (255,0,0), 1)
    cv.line(frame, (0,330), (700,330), (255,0,0), 1)

    for vehicle in vehicle_list:
        color = (0,255,0)
        if 'type' in vehicle and vehicle['type']=='heavy':
            color = (0,0,255)
        x, y, w, h = vehicle['brect']
        cx, cy = vehicle['point']
        cv.rectangle(frame, (x,y), (x+w,y+h), color, 1)
        # cv.circle(frame, (cx,cy), 2, color, -1)
        cv.putText(frame, '{}'.format(vehicle['speed']) , (cx,cy), cv.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,255), 1, cv.LINE_AA)
        text_left = 'Left: {}'.format(vehicle_count['left'])
        text_right = 'Right: {}'.format(vehicle_count['right'])
        cv.putText(frame, text_left , (230,30), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,0,0), 1, cv.LINE_AA)
        cv.putText(frame, text_right , (400,30), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,0,0), 1, cv.LINE_AA)

    return frame


def processVideo(path, vehicle_list, vehicle_count):
   
    cap = cv.VideoCapture(path)
    fgbg = cv.createBackgroundSubtractorMOG2()
    frame_count = 0

    while(1):

        ret, frame = cap.read()

        frame_count += 1
        hr, mn, sc = getTime(frame)
        time = hr*3600 + mn*60 + sc

        print("\n\nFrame id: {}, Time: {:02}:{:02}:{:02}".format(frame_count, hr, mn, sc))

        fgmask = fgbg.apply(frame)
        cv.imshow('BG-Subtraction', fgmask)

        ret, thresh = cv.threshold(fgmask, 254, 255, 0)
        cv.imshow('threshold', thresh)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        cv.imshow('Closing', closing)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        cv.imshow('Opening', opening)

        im2, contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)

        for contour in contours:    

            x, y, w, h = cv.boundingRect(contour)

            if w<25 or h<25:
                continue

            cx = x + int(w/2)
            cy = y + int(h/2)

            if cy<175:
                continue

            newveh = {
                'frameid': frame_count,
                'frames_since': 0,
                'timein': time,
                'pointin': (cx, cy),
                'time': time,
                'point': (cx, cy),
                'distance': 0,
                'brect': (x, y, w, h),
                'speed': 0,
                'counted': False
            }
            
            vehicle_list = addVehicle(newveh, vehicle_list, frame_count)

        vehicle_list, vehicle_count = updateVehicleList(vehicle_list, vehicle_count)
        for v in vehicle_list:
            print(v)
        print(vehicle_count)
        frame = draw(frame, vehicle_list, vehicle_count)

        cv.imshow('frame', frame)
        # cv.waitKey()

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

file_path = './traffic-dataset.mp4'
processVideo(file_path, vehicle_list, vehicle_count)