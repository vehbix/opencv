import cv2
import numpy as np

kamera=cv2.VideoCapture(0)
while True:
    ret,goruntu=kamera.read()
    
    hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
    
 
    dusuk_mavi=np.array([100,60,60])
    ust_mavi=np.array([140,255,255])

    mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
    son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
    
    
    kernel=np.ones((15,15),np.float32)/255
    smoothed=cv2.filter2D(son_resim,-1,kernel)
    
    blur=cv2.GaussianBlur(son_resim,(15,15),0)
    
    median=cv2.medianBlur(son_resim,15)
    
    bileteral=cv2.bilateralFilter(son_resim,15,75,75)
    
    
    # cv2.imshow("Video",goruntu)
    # cv2.imshow("mask",mask)
    cv2.imshow("son_resim",son_resim)
    # cv2.imshow("smoothed",smoothed)
    cv2.imshow("blur",blur)
    cv2.imshow("median",median)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()    