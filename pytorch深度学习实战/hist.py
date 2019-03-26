"""cv2.calcHist()，该函数有 5 个参数：
image：输入图像，传入时应该用中括号[]括起来。

channels:：传入图像的通道，如果是灰度图像，那就不用说了，只
有一个通道，值为 0，如果是彩色图像（有 3 个通道），那么值为
0,1,2 中选择一个，对应着 BGR 各个通道。这个值也得用 [] 传入。

mask：掩膜图像。如果统计整幅图，那么为 none。主要是如果要
统计部分图的直方图，就得构造相应的炎掩膜来计算。

histSize：灰度级的个数，需要中括号，比如 [256]。

ranges：像素值的范围，通常 [0,256]，有的图像如果不是 0-256，
比如说你来回各种变换导致像素值负值、很大，则需要调整后才可
以。"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('images\eye.jpg',0) #直接读为灰度图像
#opencv方法读取-cv2.calcHist（速度最快）
#图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
hist_cv = cv2.calcHist([img],[0],None,[256],[0,256])
#numpy方法读取-np.histogram()
hist_np,bins = np.histogram(img.ravel(),256,[0,256])
#numpy的另一种方法读取-np.bincount()（速度=10倍法2）
hist_np2 = np.bincount(img.ravel(),minlength=256)
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.plot(hist_cv)
plt.subplot(223),plt.plot(hist_np)
plt.subplot(224),plt.plot(hist_np2)
plt.show()

mask = np.zeros(img.shape[:2],np.uint8)
mask[100:200,100:200] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)
#opencv方法读取-cv2.calcHist（速度最快）
#图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.show()












