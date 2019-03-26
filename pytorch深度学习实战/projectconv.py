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

#定义参数
batch_size = 200
num_classes = 10
epochs = 10
#input image dimension
img_rows,img_cols = 28,28

x_train = X_train
x_test = X_test

if 'channels_first' == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_samples=x_train.shape[0]
print("num_samples:",num_samples)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=(1,1))
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=(1,1))
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1024,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if os.path.exists('mnist_torch.pkl'):
    model = torch.load('mnist_torch.pkl')
print(model)

"""#损失函数
F.log_softmax(x)
model.zero_grad()
print('conv1.bias.grad before backward')
print(model.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(model.conv1.bias,grad)

#优化算法SGD
#weight = weight -learning_rate * gradient
learning_rate = 0.01
for f in model.parameters():
    f.data.sub_(f.grad.data * learning_rate)"""

'''
trainning
'''
optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.5)
#loss=torch.nn.CrossEntropyLoss(size_average=True)
def train(epoch,x_train,y_train):
    num_batchs = num_samples// batch_size
    model.train()
    for k in range(num_batchs):
        start,end = k*batch_size,(k+1)*batch_size
        data, target = Variable(x_train[start:end],requires_grad=False),Variable(y_train[start:end],requires_grad=False)
        optimizer.zero_grad()
        output = model(data)      
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, k * len(data), num_samples,100. * k / num_samples, loss.item()))
    torch.save(model, 'mnist_torch.pkl')
'''
evaludate
'''

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    data, target = Variable(x_test, volatile=True), Variable(y_test)
    output = model(data)
    test_loss += F.nll_loss(output, target).item()
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()

    #test_loss /= len(x_test) # loss function already averages over batch size
    
    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(x_test),100. * correct / len(x_test)))

x_train=torch.from_numpy(x_train).float()
x_test=torch.from_numpy(x_test).float()
y_train=torch.from_numpy(y_train).long()
y_test=torch.from_numpy(y_test).long()
for epoch in range(1,epochs):
    train(epoch,x_train,y_train)
    test(epoch)









