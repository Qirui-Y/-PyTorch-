"""
角点的定义和特性：
角点：是一类含有足够信息且能从当前帧和下一帧中都能提取出来的点。
典型的角点检测算法：Harris 角点检测、CSS 角点检测。
好的角点检测算法的特点：1、检测出图像中“真实的”角点；2、准
确的定位性能；3、很高的重复检测率（稳定性好）；4、具有对噪
声的鲁棒性；5、具有较高的计算效率。
Open 中的函数 cv2.cornerHarris(src, blockSize, ksize, k[, dst[,
borderType]]) → dst 可以用来进行角点检测。参数如下：
src – 数据类型为 float32 的输入图像。
dst – 存储角点数组的输出图像，和输入图像大小相等。
blockSize – 角点检测中要考虑的领域大小。
ksize – Sobel 求导中使用的窗口大小。
k – Harris 角点检测方程中的自由参数，取值参数为 [0,04，0.06]。
borderType – 边界类型。
"""
# coding=utf-8
import cv2
import numpy as np
'''Harris算法角点特征提取'''
img = cv2.imread('images\eye.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# {标记点大小，敏感度（3~31,越小越敏感）}
# OpenCV函数cv2.cornerHarris() 有四个参数 其作用分别为 :
dst = cv2.cornerHarris(gray,2,23,0.04)
img[dst>0.01 * dst.max()] = [0,0,255]
cv2.imshow('corners',img)
cv2.waitKey()
cv2.destroyAllWindows()







