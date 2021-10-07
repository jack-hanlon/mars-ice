# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :Computes cumulative sum using PyCUDA inclusiveScanKernel
# Filename   :emc_gpu.py
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
import math
from time import time




if __name__ == '__main__':
    cum_sum = scan.InclusiveScanKernel(np.float32, "a+b")

    mod_clean_cum_sum = SourceModule("""
        #include <pycuda-complex.hpp>

        #include <cuComplex.h>
        #include <stdio.h>
        typedef pycuda::complex<float> cmplx;

        __global__ void clean_cum_sum(float *A, int tmp) {
            // subtract the sum of previous photons from the next as cumulative sum summed the entire array
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            tmp = i/1024;
            if(tmp != 0){
                A[i] -= A[1024*tmp -1];

            }
        }


    """)
    t3 = time()

    PHOTON_LENGTH = 1024
    num_photons = 2
    n = num_photons*PHOTON_LENGTH
    tmp = 0
    E_totsq = []
    t5 = time()
    for i in range(0,n):
        E_totsq.append(100*np.random.random_sample())
    E_totsq = np.array(E_totsq)
    print(E_totsq)
    t6 = time()
    E_totsq_gpu = gpuarray.to_gpu(E_totsq.astype(np.float32))

    #print(E_totsq)


    t1 = time()
    full_cum_sum = cum_sum(E_totsq_gpu)
    clean_cum_sum = mod_clean_cum_sum.get_function("clean_cum_sum")
    clean_cum_sum(full_cum_sum,np.int32(tmp),block=(PHOTON_LENGTH, 1, 1),grid=(num_photons,1))
    t2 = time()
    fc = full_cum_sum.get()

    for i in range(0,len(fc)):
        print(fc[i])
    t4 = time()
    print ('total time to compute cumulative sum: %f(s)'  % (t2 - t1))
    print ('total time to compute on GPU: %f(s)'  % (t4 - t3))
    print ('total time to write random E_totsq array: %f(s)'  % (t6 - t5))
