# -------------------------- #
# Author     :Jack Hanlon
# Purpose    :tets the raw module which is what we need as it is equivalent to SourceModule in PyCUDA
# Filename   :raw_module.py
# Due        :Nov 30th 2021
# ---------------------------- #
import numpy as np
import cupy as cp

loaded_from_source = r'''
extern "C"{

__global__ void test_sum(const float* x1, const float* x2, float* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}'''
module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
ker_times = module.get_function('test_multiply')
N = 10
x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.float32)
ker_sum((N,), (N,), (x1, x2, y, N**2))   # y = x1 + x2
assert cp.allclose(y, x1 + x2)
ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
assert cp.allclose(y, x1 * x2)
