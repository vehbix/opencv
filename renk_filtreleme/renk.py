import cv2
import numpy as np

kamera=cv2.VideoCapture(0)
while True:
    ret,goruntu=kamera.read()
    
    hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
    
    dusuk_kirmizi=np.array([150,3,30])
    ust_kirmizi=np.array([190,255,255])
    
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    
    dusuk_mavi=np.array([100,60,60])
    ust_mavi=np.array([140,255,255])
    
    dusuk_beyaz=np.array([0,0,140])
    ust_beyaz=np.array([256,60,256])
    
    dusuk_sarı=np.array([5,100,100])
    ust_sarı=np.array([40,255,255])
    
    mask=cv2.inRange(hsv,dusuk_beyaz,ust_beyaz)
    son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
    
    
    
    cv2.imshow("Video",goruntu)
    cv2.imshow("mask",mask)
    cv2.imshow("son_resim",son_resim)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()    