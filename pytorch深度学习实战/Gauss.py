from __future__ import print_function
import os
import struct
import numpy as np
import cv2

KSIZE = 3
SIGMA = 3
image = cv2.imread("images\eye.jpg")
print("image shape:",image.shape)
dst = cv2.GaussianBlur(image, (KSIZE,KSIZE), SIGMA, KSIZE)
cv2.imshow("img1",image)
cv2.imshow("img2",dst)
cv2.waitKey()













