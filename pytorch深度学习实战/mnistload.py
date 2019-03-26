from __future__ import print_function
import torch
import os
import struct
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    #os.path.join()函数用于路径拼接文件路径
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

X_train, y_train = load_mnist('数据集', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('数据集', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

显示图片
image1 = X_train[1]
image1 = image1.astype('float32')
image1 = image1.reshape(28,28)
cv2.imwrite('1.jpg',image1)

"""
#调用Keras方法下载
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
#调用numpy来加载mnist.npz
class mnist_data(object):
    def load_npz(self,path):
        f = np.load(path)
        for i in f:
            print i
        x_train = f['trainInps']
        y_train = f['trainTargs']
        x_test = f['testInps']
        y_test = f['testTargs']
        f.close()
        return (x_train, y_train), (x_test, y_test)
a = mnist_data()
(x_train, y_train), (x_test, y_test) = a.load_npz('D:/AI/torch/data/mnist.npz')
print ("train rows:%d,test rows:%d"% (x_train.shape[0], x_test.shape[0]))
print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape )
tt = x_train[1]
tt = tt.astype('float32')
image = tt.reshape（28,28）
cv2.imwrite("001.png,",image)
print("tt label:",y_train)

"""


"""显示十张数字图片
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()"""

















