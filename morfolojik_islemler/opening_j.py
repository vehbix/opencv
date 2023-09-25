import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Source/Resim/j.png")
img2 = cv2.imread("Source/Resim/j2.png")
img3 = cv2.imread("Source/Resim/j3.png")


kernel=np.ones((3,3),np.uint8)
erosion=cv2.erode(img2,kernel,iterations=1)
dilation=cv2.dilate(img3,kernel,iterations=1)

plt.subplot(321),plt.imshow(img),plt.title("orjinal")
plt.xticks([]),plt.yticks([])
plt.subplot(322),plt.imshow(img),plt.title("orjinal")
plt.xticks([]),plt.yticks([])
plt.subplot(323),plt.imshow(img2),plt.title("img2")
plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(erosion),plt.title("erosion")
plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(img3),plt.title("img3")
plt.xticks([]),plt.yticks([])
plt.subplot(326),plt.imshow(dilation),plt.title("dilation")
plt.xticks([]),plt.yticks([])

plt.show()