import cv2
import numpy as np
from matplotlib import pyplot as plt


messi="Source/Resim/messi.jpg"
resim=cv2.imread(messi,0)


kenarlar=cv2.Canny(resim,100,250)




plt.subplot(121),plt.imshow(resim,cmap="gray")
plt.title("orjinal"),plt.xticks([]),plt.yticks([])

plt.subplot(122),plt.imshow(kenarlar,cmap="gray")
plt.title("kenarlar"),plt.xticks([]),plt.yticks([])



plt.show()




