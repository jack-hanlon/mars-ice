import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

import numpy as np
import matplotlib.pyplot as plt
import cmath

# Not parallelized because only used once
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
# Need to parallelize
def initialize_E_curr():
    '''
    Initializes the initial vector
    '''
    E_curr = np.array([np.random.random()+np.random.random() * 1j,np.random.random()+np.random.random() * 1j])
    Norm = np.sqrt(E_curr[0]*np.conjugate(E_curr[0]) + E_curr[1]*np.conjugate(E_curr[1]))
    E_curr = E_curr/Norm
    return E_curr
# Need to parallelize
def normalize_E_next(E_next):
    '''
    Normalizes current vector
    '''
    Norm = np.sqrt(E_next[0]*np.conjugate(E_next[0]) + E_next[1]*np.conjugate(E_next[1]))
    E_next = E_next/Norm
    return E_next

#------------------------------------------------------#
#TEMPORARY CODE UNTIL PARALLELIZED#
#------------------------------------------------------#
#Initialize the Matrix operator and the initial E_curr
E_curr = initialize_E_curr()
A = random_complex_matrix()

E_next = []
for j in range(0,5):
    E_next.append(np.matmul(A,E_curr))
    E_next[j] = normalize_E_next(E_next[j])
    E_curr = E_next[j]


print(A)
print("\n")
print(E_curr)
print("\n")
print(E_next)
