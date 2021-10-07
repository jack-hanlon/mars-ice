import numpy as np
import cupy as cp


x_cpu = np.array([1,2,3])

x_gpu = cp.array([1,2,3])

l2_cpu = np.linalg.norm(x_cpu)

l2_gpu = cp.linalg.norm(x_gpu)

print(l2_cpu)
print(l2_gpu)
