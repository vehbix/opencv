import cv2
from cv2 import imread
import numpy as np
img=imread("Source/Resim/r2.jpg")
body_cascade=cv2.CascadeClassifier("Source/Html/haarcascade_fullbody.xml")

griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
bodies=body_cascade.detectMultiScale(griton,1.8,2)

for (x,y,w,h) in bodies:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    
cv2.imshow("detect",img)

cv2.waitKey(0)
cv2.destroyAllWindows()