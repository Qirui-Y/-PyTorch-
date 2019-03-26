import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.autograd import Variable
from numpy import random
x = np.linspace(-1,1,200)
y = 0.5*x+0.2*np.random.normal(0,0.05,(200,))


#将x,y转化为200 batch大小，1维度的数据
x = Variable(torch.tensor(x.reshape(200,1)))
y = Variable(torch.tensor(y.reshape(200,1)))
#print(x)

#神经网络结构
model = torch.nn.Sequential(torch.nn.Linear(1,1))
optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
loss_function = torch.nn.MSELoss()

for i in range(300):
    prediction = model(x.float())
    loss = loss_function(prediction,y.float())
    #此处有报错Expected object of scalar type Float 
    # but got scalar type Double for argument #4 'mat1'
    #需要在x后加上.float（）即可
    print("loss:" ,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(prediction.data.numpy())
plt.figure(1,figsize=(10,3))
plt.subplot(131)
plt.title('model')
plt.scatter(x.data.numpy(),y.data.numpy())
plt.plot(x.data.numpy(),y.data.numpy(),'r-',lw=5)
plt.show()


