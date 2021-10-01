# -------------------------- #
# Author     :Jack Hanlon
# Class      :CSC 497
# Purpose    :Computes EMC photons using CPU multiprocessing
# Filename   :emc_cpu_multi.py
# Due        :April 27th 2021
# ---------------------------- #
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cmath
import csv
import multiprocessing as mp
import sys
# STEP 0: FUNCTION DECLARATION
# -----------------------------------------------------------------------------#
def photon(E,photon_m,photon_n,photon_s,E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta,E_iter,m_iter,n_iter,s_iter,scatterings):
    '''
    MAIN FUNCTION (runs parallelized multiprocessed photon iteration)
    '''
    # Time taken to do N scatterings of 1 photon
    t1 = time()
    for i in range(0,scatterings):
        # STEP 2: MONTE CARLO (PICK NEW SCATTERED DIRECTION) (STEP produces a value for theta and phi used in the update step)
        # -----------------------------------------------------------------------------#
        E_totsq = []
        S = 0
        S,E_totsq,E_totsq_cumsum,index = monte_carlo(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,E_totsq,S,E,phi,theta)

        # Let s_theta and s_phi be the rotations applied to s to orient s' in the current lcsys
        s_phi = phi[index]
        s_theta = theta[index]


        # STEP 3: UPDATES TO E and Photon lcsys
        # -----------------------------------------------------------------------------#

        # The new field is:
        E = E_next_update(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta,E_scatter_theta_rot,index,E)

        # The new photon lcsys is:
        mns = np.array([photon_m,photon_n,photon_s]).T


        mprime_lc,nprime_lc,sprime_lc = photon_lcsys_update(photon_m,photon_n,photon_s,s_theta,s_phi)

        photon_m = mns*mprime_lc
        photon_n = mns*nprime_lc
        photon_s = mns*sprime_lc


        tmp_m = []
        tmp_n = []
        tmp_s = []
        for i in range(0,len(mns)):
            tmp_m.append(photon_m[i][0]+photon_m[i][1]+photon_m[i][2])
            tmp_n.append(photon_n[i][0]+photon_n[i][1]+photon_n[i][2])
            tmp_s.append(photon_s[i][0]+photon_s[i][1]+photon_s[i][2])

        photon_m = tmp_m
        photon_n = tmp_n
        photon_s = tmp_s
        photon_m = np.array(photon_m)
        photon_n = np.array(photon_n)
        photon_s = np.array(photon_s)


        E_iter.append(np.array(E))
        m_iter.append(photon_m)
        n_iter.append(photon_n)
        s_iter.append(photon_s)

    t2 = time()

    # STEP 4: Output
    # -----------------------------------------------------------------------------#
    complete = []
    for i in range(0,5):
        complete.append([E_iter[i],m_iter[i],n_iter[i],s_iter[i],i])
    print ('total time to compute scatterings for this photon on CPU: %f (s)' % (t2 - t1))
    return complete



def photon_lcsys_update(photon_m,photon_n,photon_s,theta,phi):
    '''
    Applies update matrix to vector of m,n,s
    '''
    A = np.array([[cmath.cos(theta)*cmath.cos(phi),cmath.cos(theta)*cmath.sin(phi),-1*cmath.sin(theta)],[-1*cmath.sin(phi),cmath.cos(phi),0],[cmath.sin(theta)*cmath.cos(phi),cmath.sin(theta)*cmath.sin(phi),cmath.cos(theta)]])

    mprime_lc = A[0]
    nprime_lc = A[1]
    sprime_lc = A[2]

    photon_m = mprime_lc
    photon_n = nprime_lc
    photon_s = sprime_lc

    return photon_m,photon_n,photon_s

def monte_carlo(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,E_totsq,S,E,phi,theta):
    '''
    Computes Monte Carlo approximation of scattered Electric field direction
    '''
    # PRIVATE FUNCTION (Converted from CANADIAN SPACE AGENCY PSEUDOCODE)

    return S,E_totsq,E_totsq_cumsum,index

def normalize_E_next(E_next):
    '''
    Normalizes current complex 2x1 Electric field vector
    '''
    Norm = np.sqrt(E_next[0]*np.conjugate(E_next[0]) + E_next[1]*np.conjugate(E_next[1]))
    E_next = E_next/Norm
    return E_next

def E_next_update(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta,E_scatter_theta_rot,index,E):
    '''
    Applies update to scattered Electric field using monte_carlo electric field direction phi,theta
    '''
    EE1 = [E_scatter_phi[index],E_scatter_phi_rot[index]]
    EE2 = [E_scatter_theta[index],E_scatter_theta_rot[index]]
    nE1 = E[0]*EE2[0] + E[1]*EE2[1]
    nE2 = E[0]*EE1[0] + E[1]*EE1[1]
    E[0] = nE1
    E[1] = nE2
    E = normalize_E_next(E)
    return E

def str_to_complex(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta):
    '''
    Converts list of strings into list of complex numbers and floats (theta and phi)
    '''
    E_scatter_phi = [complex(x) for x in E_scatter_phi]
    E_scatter_phi_rot = [complex(x) for x in E_scatter_phi_rot]
    E_scatter_theta_rot = [complex(x) for x in E_scatter_theta_rot]
    E_scatter_theta = [complex(x) for x in E_scatter_theta]
    phi = [float(x) for x in phi]
    theta = [float(x) for x in theta]
    return E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta

def output(E_iter_val,m_iter_val,n_iter_val,s_iter_val,i):
    '''
    Outputs to terminal first 5 scatterings and the time taken to compute total scatterings (1 photon)
    '''
    print("Scattering %d" %i)
    print("E:")
    print(E_iter_val)
    print("m:")
    print(m_iter_val)
    print("n:")
    print(n_iter_val)
    print("s:")
    print(s_iter_val)

def collect_result(result):
    '''
    Collects processes results when complete
    '''
    global results
    results.append(result)


if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print("Missing Arguments -> Enter: emc_cpu_multi.py #scatterings #photons")
        exit()
    scatterings = sys.argv[1]
    photons = sys.argv[2]
    # Guaranteeing number of scatterings is a positive integer
    try:
        scatterings = int(scatterings)
    except ValueError:
        print("Number of scatterings must be a positive integer")
        exit()

    if(scatterings <= 0):
        print("Number of scatterings must be a positive integer")
        exit()
    # Guaranteeing number of photons is a positive integer
    try:
        photons = int(photons)
    except ValueError:
        print("Number of photons must be a positive integer")
        exit()

    if(photons <= 0):
        print("Number of photons must be a positive integer")
        exit()
    # STEP 1: INITIALIZATION
    # -----------------------------------------------------------------------------#
    # Required for multiprogramming async
    results = []
    # initialize pool for # of processors in cpu (8 in this case)
    pool = mp.Pool(mp.cpu_count())
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

    #Local co-ordinate system M


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
    E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta = str_to_complex(E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta)

    # OUTPUT LISTS
    E_iter = [np.array(E)] # Electric field after each iteration of photon scattering

    m_iter = [photon_m] # photon m basis lcsys vector after each iteration of photon scattering
    n_iter = [photon_n] # photon n basis lcsys vector after each iteration of photon scattering
    s_iter = [photon_s] # photon s basis lcsys vector after each iteration of photon scattering
    t3 = time()
    for i in range(0,photons):
        pool.apply_async(photon,args=(E,photon_m,photon_n,photon_s,E_scatter_phi,E_scatter_phi_rot,E_scatter_theta_rot,E_scatter_theta,phi,theta,E_iter,m_iter,n_iter,s_iter,scatterings),callback = collect_result)
    pool.close()
    pool.join()
    t4 = time()
    print ('total time to compute scatterings for all photons on CPU: %f (s)' % (t4 - t3))
