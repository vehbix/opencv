import numpy as np
import cv2

cap=cv2.VideoCapture("Source/Video/video2.mp4")
fgbg=cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame=cap.read()
    fgmask=fgbg.apply(frame)
    cv2.imshow("fgmask",fgmask)
    cv2.imshow("orjinal",frame)
    k=cv2.waitKey(25) &0xFF
    if k==ord("q"):
        break
cap.release()
