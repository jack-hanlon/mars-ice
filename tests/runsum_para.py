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
   __global__ void running_sum(float *in,float *out,float *R, int num_photons,int block_size,float *run_sum) {
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            // Create a running sum total in one location to save memory (and to not have to sum entire arrays)

                for(int k = 0; k < block_size; k++){
                    if(run_sum[i] > R[i]){
                        out[i] = k-1;
                        break;
                    }else{
                    run_sum[i] += in[k+(block_size*i)];
                    }

                }
    }


""")
if __name__ == '__main__':

    #let R = 36 for test purposes
    t3 = time()
    #n = 1024
    block_size = 1024
    num_photons = 2
    #A = np.tile(np.arange(block_size),num_photons)
    A = np.arange(2048)
    sum_1 = 523776.0*0.2
    sum_2 = 1572352.0*0.5
    #print("sum:")
    #print(sum)
    R = [sum_1,sum_2]
    print("R:")
    #per = 0.0
    '''
    for i in range(0,num_photons):
        if(per > 0.8):
            per = 0.1
            R.append(sum_1*per)
            print("wassup")
        else:
            per += 0.1
            R.append(sum_1*per)
    '''

    R = np.array(R)
    #print("Random number val:")
    for i in range(0,len(R)):
        print(R[i])
    R_gpu = gpuarray.to_gpu(R.astype(np.float32))
    A_gpu = gpuarray.to_gpu(A.astype(np.float32))
    run_sum = np.zeros(num_photons)
    run_sum_gpu = gpuarray.to_gpu(run_sum.astype(np.float32))
    #index = 0
    out = np.zeros_like(R_gpu)
    out_gpu = gpuarray.to_gpu(out.astype(np.float32))
    t4 = time()



    t1 = time()
    running_sum = mod.get_function("running_sum")
    running_sum(A_gpu,out_gpu,R_gpu,np.int32(num_photons),np.int32(block_size),run_sum_gpu,block=(1, 1, 1),grid=(num_photons,1))
    t2 = time()
    out = out_gpu.get()
    #A_gpu.gpudata.free()
    #R_gpu.gpudata.free()
    #out_gpu.gpudata.free()
    for i in range(0,len(out)):
        print("Photon index:")
        print(out[i])
    t6 = time()
    print("Time to compute initialization: %f"% (t4-t3))
    print("Time to compute running sum: %f"% (t2-t1))
    print("Time to run entire file: %f"% (t6-t3))
