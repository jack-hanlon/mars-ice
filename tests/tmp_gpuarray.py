# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :Tests if memory can be stored in a temporary array in a PyCUDA SourceModule
# Filename   :tmp_gpuarray.py
# Due        :May 27th 2021
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
import math
from time import time


# Memory usage is becoming a bottleneck so can we reduce the memory usage by creating temporary arrays in the SourceModule


A = np.arange(100)
B = np.zeros_like(A)
A_gpu = gpuarray.to_gpu(A.astype(np.float32))
B_gpu = gpuarray.to_gpu(B.astype(np.float32))


mod = SourceModule("""
#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>
typedef pycuda::complex<float> cmplx;



__global__ void tmp_array(float *A, float *B){
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    float C[100];
     C[i] = A[i];
     B[i] = C[i];

}


""")

tmp_array = mod.get_function("tmp_array")
tmp_array(A_gpu,B_gpu,block=(100,1,1),grid=(1,1))

B = B_gpu.get()
for i in range(0,len(B)):
    print(B[i])

# Yes, temporary arrays can be used without being returned
