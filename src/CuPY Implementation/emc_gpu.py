# -------------------------- #
# Author     :Jack Hanlon
# Purpose    :Compare usage of CuPy with PyCUDA for same algo
# Filename   :emc_gpu.py
# Due        :Nov 30th 2021
# ---------------------------- #
import numpy as np
import cupy as cp
import cmath
import csv
import math
from time import time


if __name__ == '__main__':

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

    for i in range(0,len(R_list)):
        if(R_list[i] >= len(E_scatter_phi)):
            del(E_scatter_phi[-1])
            del(E_scatter_phi_rot[-1])
            del(E_scatter_theta_rot[-1])
            del(E_scatter_theta[-1])
            del(phi[-1])
            del(theta[-1])
        else:
            del(E_scatter_phi[R_list[i]])
            del(E_scatter_phi_rot[R_list[i]])
            del(E_scatter_theta_rot[R_list[i]])
            del(E_scatter_theta[R_list[i]])
            del(phi[R_list[i]])
            del(theta[R_list[i]])

    t31 = time()
    print("downsized data ...")
    #-------------------------------------------------------------------------------#
