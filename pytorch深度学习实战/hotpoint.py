import matplotlib.pyplot as plt 
import torch
import numpy as np 
from scipy.ndimage import filters
import cv2

img = cv2.imread("images/eye.jpg")
#cv2.imshow('eye',img)
#cv2.waitKey(0)
h,w,_ = img.shape
xs,ys = [],[]
for i in range(100):
    mean = w*np.random.rand(),h*np.random.rand()
    a = 50 + np.random.randint(50,200)
    b = 50 + np.random.randint(50,200)
    c = (a + b)*np.random.normal()*0.2
    
    cov = [[a,c],[c,b]]
    count = 200
    x,y = np.random.multivariate_normal(mean,cov,size=count).T 
    xs.append(x)
    ys.append(y)
x = np.concatenate(xs)
y = np.concatenate(ys)

hist,_,_ = np.histogram2d(x,y,bins=(np.arange(0,w),np.arange(0,h)))
hist = hist.T
cv2.imshow('hist',hist)
cv2.waitKey(0)


heat = filters.gaussian_filter(hist, 10.0)
plt.imshow(heat);
cv2.imshow('img',img)





















