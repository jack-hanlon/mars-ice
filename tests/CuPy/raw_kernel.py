# -------------------------- #
# Author     :Jack Hanlon
# Purpose    :test the cupy equivalent of SourceModule in PyCUDA
# Filename   :raw_kernel.py
# Due        :Nov 30th 2021
# ---------------------------- #
import numpy as np
import cupy as cp



complex_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__

void mult_arr(const complex<float>* x1, const complex<float>* x2, complex<float>* y, float a){
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + a*x2[tid];
}

''','mult_arr')

x1 = cp.arange(25,dtype=cp.complex64).reshape(5,5)
x2 = 1j*cp.arange(25,dtype=cp.complex64).reshape(5,5)
y = cp.zeros((5,5),dtype=cp.complex64)
c = 3.0
complex_kernel((5,),(5,),(x1,x2,y,cp.float32(2.0))) # Layout is grids, blocks, then arguments for the RawKernel function
print(y)
