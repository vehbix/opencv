import cv2
import numpy as np

human_cascade=cv2.CascadeClassifier("Source/Html/haarcascade_fullbody.xml")

cap=cv2.VideoCapture("Source/Video/video.mp4")

while True:
    _,frame=cap.read()
    
    griton=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    insan=human_cascade.detectMultiScale(griton,1.1,4)
    
    for (x,y,w,h) in insan:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
       
       
    cv2.imshow("insanlar",frame)
    if cv2.waitKey(25) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()