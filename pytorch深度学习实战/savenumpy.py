import numpy as np 
a = np.array([1,2,3])
np.save("nm.npy",a)
a = np.load("nm.npy")
print(a)
