# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :Computes random number in Source Module
# Filename   :random_num_c.py
# Due        :April 27th 2021
# ---------------------------- #
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

PHOTON_LENGTH = 1024
num_photons = 2
n = num_photons*PHOTON_LENGTH
t3 = time()
R = []
S = []
for i in range(0,n):
    R.append(np.random.random_sample())
    if(i % 1024 == 0):
        S.append(200)
    else:
        S.append(0)
S = np.array(S)
R = np.array(R)
S_gpu = gpuarray.to_gpu(S.astype(np.float32))
R_gpu = gpuarray.to_gpu(R.astype(np.float32))
out = np.zeros_like(R)
out_gpu = gpuarray.to_gpu(out.astype(np.float32))
t4 = time()

mod_random = SourceModule("""
    #include <pycuda-complex.hpp>

    #include <cuComplex.h>
    #include <stdio.h>
    typedef pycuda::complex<float> cmplx;


    __global__ void mult_random(float *S,float *R, int n, int scatter_bit, float *out ) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        out[i] = S[i]*R[(i+scatter_bit) % n ];

    }
    """)
reset = SourceModule("""
#include <pycuda-complex.hpp>

#include <cuComplex.h>
#include <stdio.h>
typedef pycuda::complex<float> cmplx;

__global__ void reset_array(float *A) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = 0.0;

}

""")
t1 = time()
scatter_bit = 1
for i in range(0,2):
    print("Scattering %d:" % (scatter_bit))
    mult_random = mod_random.get_function("mult_random")
    mult_random(S_gpu,R_gpu,np.int32(n),np.int32(scatter_bit),out_gpu, \
    block=(PHOTON_LENGTH, 1, 1),grid=(num_photons,1))
    out = out_gpu.get()
    for i in range(0,len(out)):
        print(out[i])
    reset_array = reset.get_function("reset_array")
    reset_array(out_gpu, block=(PHOTON_LENGTH,1,1),grid=(num_photons,1))
    #out = out_gpu.get()
    #print(out)
    #out= np.zeros_like(R)
    #out_gpu = gpuarray.to_gpu(out.astype(np.float32))
    scatter_bit += 1
t2 = time()
print ('total time to compute on GPU: %f(s)'  % (t2 - t1))
print ('total time to compute malloc on GPU: %f(s)'  % (t4 - t3))
