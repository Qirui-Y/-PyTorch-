import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
sample = Variable(torch.ones(2,2))
a=torch.Tensor(2,2)
a[0,0]=0
a[0,1]=1
a[1,0]=2
a[1,1]=3
target=Variable(a)
#print(sample,target)
criterion=nn.L1Loss()
loss=criterion(sample,target)
print("L1loss:",loss)

criterion=nn.SmoothL1Loss()
loss=criterion(sample,target)
print("Smoothloss: ",loss)

criterion=nn.MSELoss()
loss=criterion(sample,target)
print("MSEloss: ",loss)

"""criterion=nn.CrossEntropyLoss()
loss=criterion(sample,target)
print("CrossEntropyLoss: ",loss)"""












