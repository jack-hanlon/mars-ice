# Mars Ice Mapper
## Using GPU Computing to model Mars subsurface ice detection with polarimetric synthetic aperture radar
# Table of Contents
* [Introduction](#introduction)
* [Technologies](#tech)
* [Setup](#setup)
* [Comments](#comments)

# Introduction
<div style="text-align: justify">This project studied GPU-based implementations of numerical linear algebra algorithms, with a particular focus on a prototype Monte Carlo algorithm to model Mars subsurface ice detection developed by the Canadian Space Agency(CSA). 
This model has already been implemented using a conventional CPU-based approach, but may benefit substantially from the parallelism of a GPU-based implementation. Domain specific expertise and guidance was provided by Dr. Etienne Boulais of the CSA. </div>

![](https://github.com/jack-hanlon/mars-ice/blob/main/img/ice_mapper.jpg)
<font size=3>This artist illustration depicts four orbiters as part of the International Mars Ice Mapper (I-MIM) mission concept. Low and to the left, an orbiter passes above the Martian surface, detecting buried water ice through a radar instrument and large reflector antenna. Circling Mars at a higher altitude are three telecommunications orbiters with one shown relaying data back to Earth.</font>

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

# Comments
Unfortunately, the full code implementation must stay private as per the request of my supervisor at the Canadian Space Agency.


