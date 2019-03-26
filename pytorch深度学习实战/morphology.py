#二值图像处理
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def expand_image(img ,value,out=None,size=10):
    if out is None:
        w,h = img.shape
        out = np.zeros((w*size,h*size),dtype = np.uint8)
    tmp = np.repeat(np.repeat(img,size,0),size,1)
    out[:,:] = np.where(tmp,value,out)
    out[::size,:] = 0
    out[:,::size] = 0
    return out

def show_image(*imgs):
    for idx ,img in enumerate(imgs,1):
        ax = plt.subplot(1,len(imgs),idx)
        plt.imshow(img,cmap="gray")
        ax.set_axis_off()

plt.subplots_adjust(0.02,0,0.98,1,0.02,0)
plt.show()














