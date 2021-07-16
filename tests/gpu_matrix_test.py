import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import cmath

# Called every iteration of E_curr so can be multiplied by new random matrix
def random_complex_matrix():
    '''
    Returns a 2x2 Matrix of random complex numbers
    '''
    A = np.zeros((2,2), dtype=complex)
    A[0][0] = np.random.random() + np.random.random() * 1j
    A[0][1] = np.random.random() + np.random.random() * 1j
    A[1][0] = np.random.random() + np.random.random() * 1j
    A[1][1] = np.random.random() + np.random.random() * 1j
    return A
# Called once per E_curr Vector
def initialize_E_curr():
    '''
    Initializes the initial vector
    '''
    E_curr = np.array([np.random.random()+np.random.random() * 1j,np.random.random()+np.random.random() * 1j])
    Norm = np.sqrt(E_curr[0]*np.conjugate(E_curr[0]) + E_curr[1]*np.conjugate(E_curr[1]))
    E_curr = E_curr/Norm
    return E_curr
#
def normalize_E_next(E_next):
    '''
    Normalizes current vector
    '''
    Norm = np.sqrt(E_next[0]*np.conjugate(E_next[0]) + E_next[1]*np.conjugate(E_next[1]))
    E_next = E_next/Norm
    return E_next

# nxm A matrix and mxp b vector
(n, m, p) = (2, 2, 1)

n = np.int32(n)
m = np.int32(m)
p = np.int32(p)

#Redefine a matrix to contain complex values
a = random_complex_matrix()


#Redefine b vector to contain complex values
b = initialize_E_curr()



c = np.zeros((n, p), dtype=complex)


#a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
#b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
#c_gpu = drv.mem_alloc(c.size * c.dtype.itemsize)

a_gpu = gpuarray.to_gpu(a.astype(np.complex64))
b_gpu = gpuarray.to_gpu(b.astype(np.complex64))
c_gpu = gpuarray.to_gpu(c.astype(np.complex64))
#drv.memcpy_htod(a_gpu, a)
#drv.memcpy_htod(b_gpu, b)

mod = SourceModule("""
    #include <pycuda-complex.hpp>
    #include <stdio.h>

    typedef pycuda::complex<float> cmplx;
    __global__ void multiply( int n, int m, int p,cmplx *a,cmplx *b, cmplx *c )
    {
        int idx = p*threadIdx.x + threadIdx.y;

            c[idx] = 0.0;
            for(int k=0; k<m; k++){
               c[idx] += a[m*threadIdx.x+k]
                        *b[threadIdx.y+k*p] ;
                        }
    }
    """)

func = mod.get_function("multiply")

#Number of iterations of Matrix operator on vector
N = 2

#Initial Values
print("****************************************")
print("Initial Values")
print ("matrix a:")
print (a)
print ("vector b:")
print (b)


for i in range(0,N):
    func(n, m, p, a_gpu, b_gpu, c_gpu, \
        block=(np.int(n), np.int(p), 1), \
        grid=(1, 1), shared=0)
        #drv.memcpy_dtoh(c, c_gpu)
    c_gpu_out = c_gpu.get()
    print("****************************************")
    print("iteration {} \n".format(i+1))
    print("A matrix: ")

    print(a_gpu)
    print("\n")
    print(c_gpu_out)
    a = random_complex_matrix()
    a_gpu = gpuarray.to_gpu(a.astype(np.complex64))
    c_gpu_out = normalize_E_next(c_gpu_out)
    b_gpu = gpuarray.to_gpu(c_gpu_out.astype(np.complex64))
    print("****************************************")
