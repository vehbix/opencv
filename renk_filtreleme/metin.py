import cv2
import numpy as np

img=cv2.imread("Source/Resim/sayfa.jpg")

griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaus=cv2.adaptiveThreshold(griton,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
gaus=cv2.pyrUp(gaus)


cv2.imshow("img",img)
cv2.imshow("gaus",gaus)

cv2.waitKey(0)
cv2.destroyAllWindows()