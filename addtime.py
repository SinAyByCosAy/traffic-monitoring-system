import time
import cv2 as cv

def getTime():
    ctime = time.localtime(time.time())
    h, m, s = ctime.tm_hour, ctime.tm_min, ctime.tm_sec
    return '{:02}:{:02}:{:02}'.format(h, m, s)

def addToVideo(path, outpath):

    cap = cv.VideoCapture(path)
    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    outpath = cv.VideoWriter(outpath, fourcc, 20, (frame_w, frame_h))

    while(1):
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        
        time_text = getTime()
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, time_text, (10,50), font, 1, (0,0,0), 2, cv.LINE_AA)
        cv.imshow('frame', frame)

        outpath.write(frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

print(getTime())

addToVideo('./traffic-2.mp4', './traffic-time.mp4')
