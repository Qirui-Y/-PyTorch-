import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
print("conv2d sample")
a = torch.ones(4,4)
x = Variable(torch.tensor(a))
x = x.view(1,1,4,4)
print("x variable:",x)
b = torch.ones(2,2)
b[0,0] = 0.1
b[0,1] = 0.2
b[1,0] = 0.3
b[1,1] = 0.4
weights = Variable(b)
weights = weights.view(1,1,2,2)
print("weights:",weights)
y = F.conv2d(x,weights,padding=0)
print("y:",y)



