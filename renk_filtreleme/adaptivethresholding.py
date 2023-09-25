import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("Source/Resim/sudoku.jpg",0)
img=cv2.medianBlur(img,5)

ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)



basliklar=["orjinal","basit th","mean th","gaussian th"]
resimler=[img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(resimler[i],"gray")
    plt.title(basliklar[i])
    plt.xticks([]),plt.yticks([])
plt.show()