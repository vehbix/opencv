import cv2
import numpy as np

kamera=cv2.VideoCapture(0)
while True:
    ret,goruntu=kamera.read()
    
    hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
    
    dusuk_mavi=np.array([100,60,60])
    ust_mavi=np.array([140,255,255])
    
    laplacian=cv2.Laplacian(goruntu,cv2.CV_64F)
    sobelX=cv2.Sobel(goruntu,cv2.CV_64F,1,0,ksize=5)
    sobelY=cv2.Sobel(goruntu,cv2.CV_64F,0,1,ksize=5)
    

    yazi=cv2.Canny(goruntu,100,120)
    
    mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
    son_resim=cv2.bitwise_and(goruntu,goruntu,mask=mask)
        
    cv2.imshow("Video",goruntu)    
    cv2.imshow("yazi",yazi)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()    