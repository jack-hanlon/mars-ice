import math

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from time import time

if __name__ == "__main__":
    n = 4096  # It is limited by RAM size of GPU
    block_size = 1024  # Maximum possible value
    grid_size = math.ceil(n / block_size / 2)  # block_size threads can perform 2 * block_size numbers (because of the algorithm implemented)

    a = np.arange(n)  # input
    print(a)
    a = a.astype(np.float32)
    b = np.zeros_like(a)  # output

    mod = SourceModule("""
    #include <pycuda-complex.hpp>

    #include <cuComplex.h>
    #include <stdio.h>
    typedef pycuda::complex<float> cmplx;


    __global__ void sum_array(float *a, int n) {
         int tid = threadIdx.x; // Create a thread id equal to the thread index
         int offset = 2 * blockIdx.x * blockDim.x; // Create an offset value equal to length of block (Algo uses half a block per 2 * number of threads in block because the sums half the number of threads needed)

        for (int s = 1; s <= blockDim.x; s <<= 1) {
            if (tid % s == 0) {
                int idx = 2 * tid + offset;
                if (idx + s < n) {
                        if(idx == offset && idx + s == 1024 + offset){
                            continue;
                        }else{
                    atomicAdd(a + idx, a[idx + s]);
                    }
                }
            }
            __syncthreads();
        }
        
    }
    """)
    sum_array = mod.get_function("sum_array")
    t1 = time()
    a_gpu = drv.mem_alloc(a.nbytes)
    #print(a.nbytes)
    drv.memcpy_htod(a_gpu, a)
    t5 = time()
    sum_array(a_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
    t6 = time()

    drv.memcpy_dtoh(b, a_gpu)
    t2 = time()
    print("GPU result:")
    for i in range(0,len(b)):
        if(1024*i >= len(b)):
            continue
        print("b[%d]:"%(1024*i))
        print(b[1024*i])

    t3 = time()
    print("CPU result:", a.sum())
    t4 = time()

    print("GPU: %f(s)" % (t2-t1))
    print("CPU: %f(s)" % (t4-t3))
    print("GPU (w/o malloc): %f(s)" % (t6-t5))
