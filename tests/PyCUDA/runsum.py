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

mod = SourceModule("""
   __global__ void scan(float *in,float *out,float *R, int num_photons,int block_size,float run_sum) {
            const int n = num_photons*block_size;
            // Create a running sum total in one location to save memory (and to not have to sum entire arrays)
            for(int j = 0;j < num_photons; j++){
                for(int k = 0; k < n; k++){
                    if(run_sum > R[j]){
                        out[j] = k-1;
                        break;
                    }else{
                    run_sum += in[k+(j*block_size)];
                    }

                }
                run_sum = 0.0;
            }

            //__syncthreads();
    }


""")
if __name__ == '__main__':

    #let R = 36 for test purposes
    t3 = time()
    #n = 1024
    block_size = 1024
    num_photons = 1000
    A = np.tile(np.arange(block_size),num_photons)
    sum = np.sum(A)/num_photons
    R = []
    percent = 0.0
    for i in range(0,num_photons):
        percent += 0.1
        R.append(sum*percent)

    R = np.array(R)
    #print(R)
    R_gpu = gpuarray.to_gpu(R.astype(np.float32))
    A_gpu = gpuarray.to_gpu(A.astype(np.float32))
    run_sum = 0.0
    #index = 0
    out = np.zeros_like(R_gpu)
    out_gpu = gpuarray.to_gpu(out.astype(np.float32))
    t4 = time()



    t1 = time()
    scan = mod.get_function("scan")
    scan(A_gpu,out_gpu,R_gpu,np.int32(num_photons),np.int32(block_size),np.float32(run_sum),block=(block_size, 1, 1),grid=(num_photons,1))
    t2 = time()
    out = out_gpu.get()
    A_gpu.gpudata.free()
    R_gpu.gpudata.free()
    out_gpu.gpudata.free()
    for i in out:
        print("Photon index:")
        print(i)
    t6 = time()
    print("Time to compute initialization: %f"% (t4-t3))
    print("Time to compute running sum: %f"% (t2-t1))
    print("Time to run entire file: %f"% (t6-t3))
