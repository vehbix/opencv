import numpy as np
import cv2
import matplotlib.pyplot as plt

resim_aranacak=cv2.imread("Source/Resim/kucuk_resim.jpg")
resim_buyuk=cv2.imread("Source/Resim/buyuk_resim.jpg")

orb=cv2.ORB_create()
anahtar1,hedef1=orb.detectAndCompute(resim_aranacak,None)
anahtar2,hedef2=orb.detectAndCompute(resim_buyuk,None)
burutal_force=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
eslesmeler=burutal_force.match(hedef1,hedef2)
eslesmeler=sorted(eslesmeler,key=lambda x:x.distance)
son_resim=cv2.drawMatches(resim_aranacak,anahtar1,resim_buyuk,anahtar2,eslesmeler[:10],None,flags=2)
plt.imshow(son_resim)
plt.show()