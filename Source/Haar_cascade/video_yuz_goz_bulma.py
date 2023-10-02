import cv2
import numpy as np
yuz_cascade=cv2.CascadeClassifier("Source/Html/haarcascade_frontalface_default.xml")
goz_cascade=cv2.CascadeClassifier("Source/Html/haarcascade_eye.xml")

kamera=cv2.VideoCapture(0)

while(kamera.isOpened()):
    ret,frame=kamera.read()
    
    griton=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    yuzler=yuz_cascade.detectMultiScale(griton,1.2,4)
    
    for (x,y,w,h) in yuzler:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        roi_griton=griton[y:y+h,x:x+w]
        roi_renkli=frame[y:y+h,x:x+w]
        gozler=goz_cascade.detectMultiScale(roi_griton,1.6,4)
        for(ex,ey,ew,eh) in gozler:
            cv2.rectangle(roi_renkli,(ex,ey),(ex+ew,ey+eh),(125,0,255),2)
    
       
    cv2.imshow("orjinal",frame)
    if cv2.waitKey(25) & 0xFF==ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()