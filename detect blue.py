import cv2 as cv
import numpy as np


vid = cv.VideoCapture(0)

while 1:
    r,frame = vid.read()
    hsv = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    hsv = cv.medianBlur(hsv,13)

    lowerBlue = np.array([0,70,150])
    upperBlue = np.array([121,255,255])
    mask = cv.inRange(hsv,lowerBlue,upperBlue)

    cnt,hie = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)


    for cn in cnt:
        area = cv.contourArea(cn)
        if area>1000:
            cv.drawContours(frame,cn,-1,(0,255,0),3)


    cv.imshow("pencere",mask)
    cv.imshow("fs",frame)
    if cv.waitKey(1) & 0xFF==ord("q"):
        break

vid.release()
cv.destroyAllWindows()