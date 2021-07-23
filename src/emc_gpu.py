# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :Computes EMC photons using CUDA programming
# Filename   :emc_gpu.py
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
# Step 0: FUNCTION DECLARATION
# -----------------------------------------------------------------------------#

if __name__ == '__main__':
    t9 = time()
    t40 = time()
    print("File started: ")

    mod = SourceModule("""
    #include <pycuda-complex.hpp>
    #include <math.h>
    #include <cuComplex.h>
    #include <stdio.h>
    typedef pycuda::complex<float> cmplx;


    /*----------------------------------------------------------------------------------------------------------------*/
    __global__ void monte_carlo(cmplx *E_scatter_phi, cmplx *E_scatter_phi_rot,cmplx *E_scatter_theta_rot,cmplx *E_scatter_theta, cmplx *theta, cmplx *E, cmplx *E_totsq,float *float_E_totsq, float *cum_sum_E_totsq){

      // each block will have one photon doing this our gpuarray will be length of all photons and be split up accordingly
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      // PRIVATE FUNCTION (Converted from CANADIAN SPACE AGENCY PSEUDOCODE)


    }
    /*----------------------------------------------------------------------------------------------------------------*/
    __global__ void sum_array(float *a, int n) {
         int tid = threadIdx.x; // Create a thread id equal to the thread index
         int offset = 2 * blockIdx.x * blockDim.x; // Create an offset value equal to length of block (Algo uses half a block per 2 * number of threads in block because the sums half the number of threads needed)

        // Compute the sum of E_totsq for each photon
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
    /*----------------------------------------------------------------------------------------------------------------*/
    __global__ void mult_random(float *S,float *R, int n, int scatter_bit, float *out,float *out_reduced ) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        // On each scattering - reset the out and out_reduced arrays to 0
        out[i] = 0;
        __syncthreads();
        out_reduced[i/1024] = 0;
        __syncthreads();
        out[i] = S[i]*R[(i+scatter_bit) % n ];
        __syncthreads();

         // Remove zeros from out_gpu (array of vals = sum of E_totsq * random number)
         if(i % 1024 == 0){
            out_reduced[i/1024] = out[i];
         }


    }
    /*----------------------------------------------------------------------------------------------------------------*/
     __global__ void running_sum(float *in,float *out,float *R, int num_photons,int block_size,float *run_sum) {
              const int i = threadIdx.x + blockIdx.x * blockDim.x;
              // Empty previous calculation from temporary arrays
              run_sum[i] = 0;
              __syncthreads();
              out[i] = 0;
              __syncthreads();
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
      /*----------------------------------------------------------------------------------------------------------------*/
     __global__ void E_next_update(float *INDEX,cmplx *E_scatter_phi, cmplx *E_scatter_phi_rot,cmplx *E_scatter_theta_rot,cmplx *E_scatter_theta,cmplx *EE1, cmplx *EE2,cmplx *nE,cmplx *Norm,cmplx *E){
            // Compute the update to the scattered electric field
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            int k = i/2;
            // Do first line of emc_cpu.py E_next_update
            if(i % 2 == 0){
                //EVEN
                EE1[i] = E_scatter_phi[int(INDEX[k])];
                EE2[i] =  E_scatter_theta[int(INDEX[k])];
            }else{
                //ODD
                EE1[i] = E_scatter_phi_rot[int(INDEX[k])];
                EE2[i] =  E_scatter_theta_rot[int(INDEX[k])];
            }

            // Do second line of emc_cpu.py E_next_update
            if(i % 2 == 0){
                //EVEN
                nE[i] = E[i]*EE2[i] + E[i+1]*EE2[i+1];
            }else{
                //ODD
                nE[i] = E[i-1]*EE1[i-1] + E[i]*EE1[i];

            }

            // Perform update to Electric field
            E[i] = nE[i];


            // Normalize the E field
            if(i % 2 == 0){
                //EVEN
                Norm[i] = sqrt(E[i]*conj(E[i]) + E[i+1]*conj(E[i+1]));

            }else{
                //ODD
                Norm[i] = Norm[i-1];
            }

            // Return E normalized.
            E[i] = nE[i]/Norm[i];



     }
     /*----------------------------------------------------------------------------------------------------------------*/
     __global__ void photon_lcsys_update(float *INDEX,float *theta,float *phi,float *photon_m, float *photon_n, float *photon_s, float *mprime_lc, float *nprime_lc, float *sprime_lc, float *tmp_m, float *tmp_n, float *tmp_s){
            // Update the photon local co-ordinate system for all photons in parallel
            const int i = threadIdx.x + blockIdx.x*blockDim.x;

            // Compute Rotation Matrix for each index
            if(i % 3 == 0){
                // First element of mprime_lc for each photon
                mprime_lc[i] = cos(theta[int(INDEX[i/3])])*cos(phi[int(INDEX[i/3])]);
                mprime_lc[i+1] = cos(theta[int(INDEX[i/3])])*sin(phi[int(INDEX[i/3])]);
                mprime_lc[i+2] = -1*sin(theta[int(INDEX[i/3])]);

                nprime_lc[i] = -1*sin(phi[int(INDEX[i/3])]);
                nprime_lc[i+1] = cos(phi[int(INDEX[i/3])]);
                nprime_lc[i+2] = 0;

                sprime_lc[i] = sin(theta[int(INDEX[i/3])])*cos(phi[int(INDEX[i/3])]);
                sprime_lc[i+1] = sin(theta[int(INDEX[i/3])])*sin(phi[int(INDEX[i/3])]);
                sprime_lc[i+2] = cos(theta[int(INDEX[i/3])]);
            }

            __syncthreads();
            // Set to zero any values returned in scientific notation
            if(abs(mprime_lc[i]) < 0.0000001){
              mprime_lc[i] = 0.0;
            }
            if(abs(nprime_lc[i]) < 0.0000001){
              nprime_lc[i] = 0.0;
            }
            if(abs(sprime_lc[i]) < 0.0000001){
              sprime_lc[i] = 0.0;
            }

            __syncthreads();

            // Logically compute a matrix transpose and perform a multiplication of elements of photon_m,n,s by m,n,sprime_lc respectively
            if(i % 3 == 0){
                tmp_m[i] = photon_m[i]*mprime_lc[i] + photon_n[i]*mprime_lc[i+1] + photon_s[i]*mprime_lc[i+2];
                tmp_m[i+1] = photon_m[i+1]*mprime_lc[i] + photon_n[i+1]*mprime_lc[i+1] + photon_s[i+1]*mprime_lc[i+2];
                tmp_m[i+2] = photon_m[i+2]*mprime_lc[i] + photon_n[i+2]*mprime_lc[i+1] + photon_s[i+2]*mprime_lc[i+2];

                tmp_n[i] = photon_m[i]*nprime_lc[i] + photon_n[i]*nprime_lc[i+1] + photon_s[i]*nprime_lc[i+2];
                tmp_n[i+1] = photon_m[i+1]*nprime_lc[i] + photon_n[i+1]*nprime_lc[i+1] + photon_s[i+1]*nprime_lc[i+2];
                tmp_n[i+2] = photon_m[i+2]*nprime_lc[i] + photon_n[i+2]*nprime_lc[i+1] + photon_s[i+2]*nprime_lc[i+2];

                tmp_s[i] = photon_m[i]*sprime_lc[i] + photon_n[i]*sprime_lc[i+1] + photon_s[i]*sprime_lc[i+2];
                tmp_s[i+1] = photon_m[i+1]*sprime_lc[i] + photon_n[i+1]*sprime_lc[i+1] + photon_s[i+1]*sprime_lc[i+2];
                tmp_s[i+2] = photon_m[i+2]*sprime_lc[i] + photon_n[i+2]*sprime_lc[i+1] + photon_s[i+2]*sprime_lc[i+2];
            }
            __syncthreads();

            // Update photon lcsys m,n,s to m',n',s'

            photon_m[i] = tmp_m[i];
            __syncthreads();
            photon_n[i] = tmp_n[i];
            __syncthreads();
            photon_s[i] = tmp_s[i];



      }
     /*----------------------------------------------------------------------------------------------------------------*/
    """)

    t41 = time()
    # STEP 1: Inititialize E, csv arrays, m,n,s
    # -----------------------------------------------------------------------------#
    # Inititialize the input data arrays
    E_scatter_phi= []
    E_scatter_phi_rot = []
    E_scatter_theta_rot = []
    E_scatter_theta = []
    phi = []
    theta = []

    # E in local co-ordinate system lcsys
    E = np.array([1+0j,0+0j])
    # local co-ordinate system of photon (initial basis)
    photon_m = np.array([1,0,0]) # points in direction of E field
    photon_n = np.array([0,1,0]) # points in direction of perpendicular to E field (B field)
    photon_s = np.array([0,0,1]) # points in direction of photon propagation


    # Read in the data from scattering.csv
    with open('scattering.csv', "r") as  file:
        csv_reader = csv.reader(file,delimiter=",")
        for lines in csv_reader:
            E_scatter_phi.append(lines[0])
            E_scatter_phi_rot.append(lines[1])
            E_scatter_theta_rot.append(lines[2])
            E_scatter_theta.append(lines[3])
            phi.append(lines[4])
            theta.append(lines[5])

    # Convert data from strings to appropriate data type
    E_scatter_phi = [complex(x) for x in E_scatter_phi]
    E_scatter_phi_rot = [complex(x) for x in E_scatter_phi_rot]
    E_scatter_theta_rot = [complex(x) for x in E_scatter_theta_rot]
    E_scatter_theta = [complex(x) for x in E_scatter_theta]
    phi = [float(x) for x in phi]
    theta = [float(x) for x in theta]

    print("imported data from csv ...")
    # DOWN SAMPLING
    #-------------------------------------------------------------------------------#
    # Divides 5248 element long lists into 1312 element lists
    E_scatter_phi = E_scatter_phi[0::4]
    E_scatter_phi_rot = E_scatter_phi_rot[0::4]
    E_scatter_theta_rot = E_scatter_theta_rot[0::4]
    E_scatter_theta = E_scatter_theta[0::4]
    phi = phi[0::4]
    theta = theta[0::4]

    # Randomly remove 288 elements from lists so they are all 1024 elements long (This is why outputs differ on each run of emc_gpu.py --> shouldn't be a problem though)
    t30 = time()
    R_list = []
    for i in range(0,288):
        R = int(1311*np.random.random_sample())
        R_list.append(R)


    R_list = sorted(R_list)
    # E_scatter_phi
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(E_scatter_phi)):
            del(E_scatter_phi[-1])
        else:
            del(E_scatter_phi[R_list[i]])
    # E_scatter_phi_rot
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(E_scatter_phi_rot)):
            del(E_scatter_phi_rot[-1])
        else:
            del(E_scatter_phi_rot[R_list[i]])
    #E_scatter_theta_rot
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(E_scatter_theta_rot)):
            del(E_scatter_theta_rot[-1])
        else:
            del(E_scatter_theta_rot[R_list[i]])
    # E_scatter_theta
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(E_scatter_theta)):
            del(E_scatter_theta[-1])
        else:
            del(E_scatter_theta[R_list[i]])
    # Phi
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(phi)):
            del(phi[-1])
        else:
            del(phi[R_list[i]])
    # theta
    for i in range(0,len(R_list)):
        if(R_list[i] >= len(theta)):
            del(theta[-1])
        else:
            del(theta[R_list[i]])

    t31 = time()
    print("downsized data ...")
    #-------------------------------------------------------------------------------#
    PHOTON_LENGTH =
    # Convert lists to np arrays
    E_scatter_phi = np.array(E_scatter_phi)
    E_scatter_phi_rot = np.array(E_scatter_phi_rot)
    E_scatter_theta_rot = np.array(E_scatter_theta_rot)
    E_scatter_theta = np.array(E_scatter_theta)
    phi = np.array(phi)
    theta = np.array(theta)

    # This value will be an input from command line value
    num_photons = 20000
    # for sum_array halves number of blocks used
    grid_sum_size = int(math.ceil(num_photons/2))
    # for sum_array calculates length of array sent to GPU kernel
    n = num_photons*PHOTON_LENGTH
    t20 = time()
    # tile the arrays to match number of photons
    photon_m = np.tile(photon_m,num_photons)
    photon_n = np.tile(photon_n,num_photons)
    photon_s = np.tile(photon_s,num_photons)

    mprime_lc = np.zeros_like(photon_m)
    nprime_lc = np.zeros_like(photon_n)
    sprime_lc = np.zeros_like(photon_s)

    tmp_m = np.zeros_like(photon_m)
    tmp_n = np.zeros_like(photon_n)
    tmp_s = np.zeros_like(photon_s)



    E = np.tile(E,num_photons)

    E_scatter_phi = np.tile(E_scatter_phi,num_photons)
    E_scatter_phi_rot = np.tile(E_scatter_phi_rot,num_photons)
    E_scatter_theta_rot = np.tile(E_scatter_theta_rot,num_photons)
    E_scatter_theta = np.tile(E_scatter_theta,num_photons)
    phi = np.tile(phi,num_photons)
    theta = np.tile(theta,num_photons)
    t21 = time()
    # Empty Electric field monte carlo calculation array
    E_totsq = np.zeros_like(theta)
    # Dtype memory needs to be allocated as atomicAdd fct does not handle complex numbers so Real part needs to be comverted to float/double
    float_E_totsq = np.zeros_like(E_totsq)
    float_E_totsq = float_E_totsq.astype(np.float32)
    # Lists that store the output E,m,n,s after each scattering



    # STEP 2: SEND FIRST BATCH OF PHOTONS TO COMPLETE SCATTERINGS ON GPU
    # -----------------------------------------------------------------------------#
    # Create a gpuarray of random numbers (length should be data length x # photons)
    t11 = time()
    R = np.random.random_sample(size = n)
    R = np.array(R)
    t12 = time()
    print("Produced array of random numbers ...")
    print("length of R %d "%(len(R)))
    #Send random number array to gpuarray
    t13 = time()
    R_gpu = gpuarray.to_gpu(R.astype(np.float32))

    # data arrays and initial electric field memory allocated to GPU

    photon_m_gpu = gpuarray.to_gpu(photon_m.astype(np.float32))
    photon_n_gpu = gpuarray.to_gpu(photon_n.astype(np.float32))
    photon_s_gpu = gpuarray.to_gpu(photon_s.astype(np.float32))

    mprime_lc_gpu = gpuarray.to_gpu(mprime_lc.astype(np.float32))
    nprime_lc_gpu = gpuarray.to_gpu(nprime_lc.astype(np.float32))
    sprime_lc_gpu = gpuarray.to_gpu(sprime_lc.astype(np.float32))

    tmp_m_gpu = gpuarray.to_gpu(tmp_m.astype(np.float32))
    tmp_n_gpu = gpuarray.to_gpu(tmp_n.astype(np.float32))
    tmp_s_gpu = gpuarray.to_gpu(tmp_s.astype(np.float32))

    E_gpu = gpuarray.to_gpu(E.astype(np.complex64))
    E_scatter_phi_gpu = gpuarray.to_gpu(E_scatter_phi.astype(np.complex64))
    E_scatter_phi_rot_gpu = gpuarray.to_gpu(E_scatter_phi_rot.astype(np.complex64))
    E_scatter_theta_rot_gpu = gpuarray.to_gpu(E_scatter_theta_rot.astype(np.complex64))
    E_scatter_theta_gpu = gpuarray.to_gpu(E_scatter_theta.astype(np.complex64))
    phi_gpu = gpuarray.to_gpu(phi.astype(np.complex64))
    theta_gpu = gpuarray.to_gpu(theta.astype(np.complex64))
    float_theta_gpu = gpuarray.to_gpu(theta.astype(np.float32))
    float_phi_gpu = gpuarray.to_gpu(phi.astype(np.float32))

    #output array sent to GPU to be filled.
    E_totsq_gpu = gpuarray.to_gpu(E_totsq.astype(np.complex64))
    # Used in the cumulative sum (Data copied from E_totsq_gpu (np.complex64) into cum_sum_E_totsq_gpu (np.float32))
    cum_sum_E_totsq = np.zeros_like(R)
    cum_sum_E_totsq_gpu = gpuarray.to_gpu(cum_sum_E_totsq.astype(np.float32))
    float_E_totsq_gpu = drv.mem_alloc(float_E_totsq.nbytes)
    drv.memcpy_htod(float_E_totsq_gpu,float_E_totsq)
    # output array for the Sum * random number calculation
    out = np.zeros_like(R)
    out_reduced = np.zeros(num_photons)
    out_gpu = gpuarray.to_gpu(out.astype(np.float32))
    out_gpu_reduced = gpuarray.to_gpu(out_reduced.astype(np.float32))
    run_sum = np.zeros(num_photons)
    run_sum_gpu = gpuarray.to_gpu(run_sum.astype(np.float32))

    INDEX = np.zeros(num_photons)
    INDEX_gpu = gpuarray.to_gpu(INDEX.astype(np.float32))
    #float_E_totsq_gpu = gpuarray.to_gpu(float_E_totsq.astype(np.float64))
    EE1 = np.zeros_like(E)
    EE2 = np.zeros_like(E)
    EE1_gpu = gpuarray.to_gpu(EE1.astype(np.complex64))
    EE2_gpu = gpuarray.to_gpu(EE2.astype(np.complex64))

    nE = np.zeros_like(E)
    Norm = np.zeros_like(E)
    nE_gpu = gpuarray.to_gpu(nE.astype(np.complex64))
    Norm_gpu = gpuarray.to_gpu(Norm.astype(np.complex64))
    t14 = time()

    print("Sent data to gpu ...")
    # Used to permute the random number array so each scattering is multiplied by a different random number
    scatter_bit = 1


    # SCATTERING LOOP WILL START HERE
    # CALCULATING E_totsq
    # ------------------------------------------------------------------------- #
    t1 = time()
    print("starting scattering loop ...")
    # PyCUDA collecting C CUDA functions for use in Python
    monte_carlo = mod.get_function("monte_carlo")
    sum_array = mod.get_function("sum_array")
    mult_random = mod.get_function("mult_random")
    running_sum = mod.get_function("running_sum")
    E_next_update = mod.get_function("E_next_update")
    photon_lcsys_update = mod.get_function("photon_lcsys_update")

    # SCATTERING LOOP (A faster implementation would have these kernels as one monolithic kernel with this for loop inside the kernel rather than on the host side)
    for i in range(0,1000):

        # Compute E_totsq (fast)
        monte_carlo(E_scatter_phi_gpu,E_scatter_phi_rot_gpu,E_scatter_theta_rot_gpu,E_scatter_theta_gpu,theta_gpu,E_gpu,E_totsq_gpu,float_E_totsq_gpu,cum_sum_E_totsq_gpu, \
        block=(PHOTON_LENGTH, 1, 1),grid=(num_photons,1))

        # SUMMING E_totsq
        # ------------------------------------------------------------------------- #

        # Compute sum of E_totsq array (fast)
        sum_array(float_E_totsq_gpu,np.int32(n),block=(PHOTON_LENGTH,1,1),grid=(grid_sum_size,1))

        # MULTIPLYING SUM OF E_totsq by RANDOM NUMBERS
        # ------------------------------------------------------------------------- #
        # multiply each sum of E_totsq for each  photon by a uniform random number
        mult_random(float_E_totsq_gpu,R_gpu,np.int32(n),np.int32(scatter_bit),out_gpu,out_gpu_reduced, \
        block=(PHOTON_LENGTH, 1, 1),grid=(num_photons,1))

        # Iterate scattering bit so each scattering has a different random number multiplication
        scatter_bit += 1

        # COMPUTE THE RUNNING SUM OF EACH PHOTON IN PARALLEL AND PICK THE INDEX FOR EACH PHOTON
        # ------------------------------------------------------------------------- #
        # Full Cumulative sum of E_totsq_gpu

        running_sum(cum_sum_E_totsq_gpu,INDEX_gpu,out_gpu_reduced,np.int32(num_photons),np.int32(PHOTON_LENGTH),run_sum_gpu,block=(1, 1, 1),grid=(num_photons,1))

        # UPDATE E , m , n and s
        # ------------------------------------------------------------------------- #


        # Update E

        E_next_update(INDEX_gpu,E_scatter_phi_gpu,E_scatter_phi_rot_gpu,E_scatter_theta_rot_gpu,E_scatter_theta_gpu,EE1_gpu,EE2_gpu,nE_gpu,Norm_gpu,E_gpu,block=(2,1,1),grid=(num_photons,1))

        # Update m, n , s to m', n', s'


        photon_lcsys_update(INDEX_gpu,float_theta_gpu,float_phi_gpu,photon_m_gpu,photon_n_gpu,photon_s_gpu,mprime_lc_gpu,nprime_lc_gpu,sprime_lc_gpu,tmp_m_gpu,tmp_n_gpu,tmp_s_gpu,block=(3,1,1),grid=(num_photons,1))




    t2 = time()
    t10 = time()
    print("EMC executed for %d photons" % (num_photons))
    print ('(t2-t1)total time to compute scatterings: %f(s)'  % (t2 - t1))
    print ('(t10-t9)total time to run entire file: %f(s)'  % (t10 - t9))
    print('(t12-t11)Time to create random number array %f' %(t12 - t11))
    print('(t14-t13)Time to memory allocate %f' %(t14 - t13))
    print('(t21-t20)Time to tile arrays %f' %(t21 - t20))
    print('(t31-t30)Time to downsize arrays %f' %(t31 - t30))
    print('(t41-t40)Time to compile SourceModule %f' %(t41 - t40))
