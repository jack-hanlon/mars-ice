# Mars Ice Mapper
## Using GPU Computing to model Martian subsurface ice detection with polarimetric synthetic aperture radar
# Table of Contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Setup](#setup)
* [Comments](#comments)
* [Bench Marks](#bench-marks)
* [What's next?](#what's-next?)


# Introduction
This project studied GPU-based implementations of numerical linear algebra algorithms, with a particular focus on a prototype Monte Carlo algorithm to model Mars subsurface ice detection developed by the Canadian Space Agency(CSA).  
  
This model has already been implemented using a conventional CPU-based approach, but may benefit substantially from the parallelism of a GPU-based implementation. Domain specific expertise and guidance was provided by Dr. Etienne Boulais of the CSA.

![](https://github.com/jack-hanlon/mars-ice/blob/main/img/ice_mapper.jpg)
This artist illustration depicts four orbiters as part of the International Mars Ice Mapper (I-MIM) mission concept. Low and to the left, an orbiter passes above the Martian surface, detecting buried water ice through a radar instrument and large reflector antenna. Circling Mars at a higher altitude are three telecommunications orbiters with one shown relaying data back to Earth.

Image Credits: NASA
https://www.nasa.gov/feature/nasa-international-partners-assess-mission-to-map-ice-on-mars-guide-science-priorities
# Technologies
Project is created with:
* Python version: 3.8.5
* PyCUDA version: 2021.1
* Windows 10
* Anaconda 3
* Visual Studio 2019
* CUDA toolkit 11.2 & 11.3

Learn how to setup a local PyCUDA environment here: https://jack-hanlon.github.io/tutorials/2021/06/21/How-to-set-up-a-PyCUDA-environment-on-Windows-10.html

# Setup
To run this project, first install a PyCUDA environment on Windows 10: https://jack-hanlon.github.io/tutorials/2021/06/21/How-to-set-up-a-PyCUDA-environment-on-Windows-10.html  
Next, clone this repo and cd into the source (src) directory.  
From here, you will see three Electric Field Monte Carlo files.  
* emc_cpu.py
* emc_cpu_multi.py
* emc_gpu.py
### emc_cpu.py
This is the original source code for the Electric field Monte Carlo algorithm, developed using pseudocode shared by the Canadian Space Agency.
### emc_cpu_multi.py
A simple runtime upgrade to the initial algorithm using the Python multiprocessing library.
### emc_gpu.py
An implementation of a hardware accelerated EMC algorithm using PyCUDA kernels.  

Run these files using the commands specified in the How to run EMC.pdf file in the mars-ice-mapper/docs folder.  
Utilize the Requirements Specification Document for understanding how the kernels work.  
# Comments
Unfortunately, the full code implementation must stay private as per the request of my supervisor at the Canadian Space Agency. However, the Monte Carlo implementation is not too complicated following the algorithm explanation in the research paper.  
This project is mainly on Github to display my experience, nevertheless it may be useful to those working on similar projects and those learning CUDA.
# Bench Marks
On a Nvidia mx 150 GPU + Intel i5-8250U CPU:
| File | emc_cpu.py   |      emc_cpu_multi.py      |  emc_gpu.py | emc_gpu.py w/ a RTX 3080 |
|----------|----------|:-------------:|------:|----------|
|Avg time to compute a photon| 1.28 s/ph |  0.60 s/ph | 0.00103 s/ph | 0.0000692 s/ph |
|Time to compute 100,000| 35.56 hrs | 16.68 hrs | 1.72 mins| 6.92 secs |

Runtime for 100,000 photons scattered 1000 times each.

# What's next?
* The emc_gpu.py file will be refactored and unit tested.
* The 6 kernel calls will be converted into 1 monolithic kernel with the scattering loop inside to remove the Python overhead.
* The file will be tested on a Nvidia RTX 8000 to test its optimal benchmark for operational use at the Canadian Space Agency.
* The file could also be converted into C/C++ and using CUDA to remove all Python overhead for a more realistic use case on orbital satellite's.
