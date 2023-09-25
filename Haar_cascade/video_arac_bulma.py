import cv2
import numpy as np

car_cascade=cv2.CascadeClassifier("Source/Html/cars.xml")
cap1=cv2.VideoCapture("Source/Video/video1.avi")
cap2=cv2.VideoCapture("Source/Video/video2.avi")

while True:
    _,frame1=cap1.read()
    _,frame2=cap2.read()
    
    griton1=cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    griton2=cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)
    
    araclar1=car_cascade.detectMultiScale(griton1,1.1,1)
    araclar2=car_cascade.detectMultiScale(griton2,1.1,1)
    
    for (x,y,w,h) in araclar1:
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,255),3)
    for (x,y,w,h) in araclar2:
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,255),3)
       
       
    cv2.imshow("video1",frame1)
    cv2.imshow("video2",frame2)
    if cv2.waitKey(25) & 0xFF==ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()