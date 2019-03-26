import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

print("conv1d sample")
a=range(16)
x = Variable(torch.Tensor(a))
x=x.view(1,1,16)
print("x variable:", x)
b=torch.ones(3)
b[0]=0.1
b[1]=0.2
b[2]=0.3
weights = Variable(b)
weights=weights.view(1,1,3)
print ("weights:",weights)
y=F.conv1d(x, weights, padding=0)
print ("y:",y)




