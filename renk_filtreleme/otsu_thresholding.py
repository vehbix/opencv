import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Source/Resim/gurultuluresim.jpg",0)

_,th1=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
_,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur=cv2.GaussianBlur(img,(9,9),0)
_,thblur=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

resimler=[img,0,th1,
          img,0,th2,
          blur,0,thblur]
basliklar=["orjinal","histogram","basit thresholding",
           "orjinal","histogram","otsu thresholding",
           "guassian blur","histogram","otsu thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(resimler[i*3],"gray")
    plt.title(basliklar[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(resimler[i*3].ravel(),256)
    plt.title(basliklar[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(resimler[i*3+2],"gray")
    plt.title(basliklar[i*3+2]),plt.xticks([]),plt.yticks([])



plt.show()