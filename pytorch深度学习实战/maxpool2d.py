import torch 
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn

print("conv2d  sample")
a = range(20)
x = Variable(torch.Tensor(a))
x = x.view(1,1,4,5)
print("x variable:",x)
y = F.max_pool2d(x,kernel_size=2,stride=2)
print("y:",y)













