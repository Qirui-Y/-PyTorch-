#膨胀和腐蚀
from scipy.ndimage import morphology
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
def dilation_demo(a, structure=None):
    b = morphology.binary_dilation(a, structure)
    img = expand_image(a, 255)
    return expand_image(np.logical_xor(a,b), 150, out=img)

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

a = plt.imread("images\eye.jpg")[:,:,0].astype(np.uint8)
img1 = expand_image(a, 255)
img2 = dilation_demo(a)
img3 = dilation_demo(a, [[1,1,1],[1,1,1],[1,1,1]])
show_image(img1, img2, img3)

