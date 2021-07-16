# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :tests sync order of threads
# Filename   :sync_test.py
# Due        :April 27th 2021
# ---------------------------- #
#PYCUDA IMPORTS
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda import cumath
from pycuda import curandom
from pycuda import scan

# PYTHON IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import cmath
import csv
from time import time

A = []
B = []
for i in range(0,1024):
    A.append(i)
    B.append(i+1024)
print(A)
print("b:")
print(B)
A = np.array(A)
B = np.array(B)

C = np.concatenate((A,B),axis=None)
out = np.zeros_like(C)
C_gpu = gpuarray.to_gpu(C)
out_gpu = gpuarray.to_gpu(out)
E = [1,0]
E = np.array(E)
E_gpu = gpuarray.to_gpu(E)

mod = SourceModule("""
#include <stdlib.h>
#include <stdio.h>

__global__ void multiply(int *E,int *C, int *out)
{

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  out[i] += C[i]*E[0] + C[i]*E[0];
  __syncthreads();

}


""")


mult = mod.get_function("multiply")

mult(E_gpu,C_gpu,out_gpu, \
block=(1024, 1, 1),grid=(2,1))


out = out_gpu.get()
for i in out:
    print(i)
