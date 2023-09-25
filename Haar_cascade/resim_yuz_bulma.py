import cv2
import numpy as np

img=cv2.imread("Source/Resim/kalabalik.jpg")
yuz_casc=cv2.CascadeClassifier("Source/Html/haarcascade_frontalface_default.xml")

griton=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
yuzler=yuz_casc.detectMultiScale(griton,1.2,2)

for (x,y,w,h) in yuzler:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    
cv2.imshow("detect",img)

cv2.waitKey(0)
cv2.destroyAllWindows()