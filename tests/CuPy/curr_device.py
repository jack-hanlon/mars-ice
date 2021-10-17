# -------------------------- #
# Author     :Jack Hanlon
# Purpose    :current device test
# Filename   :curr_device.py
# Due        :Nov 30th 2021
# ---------------------------- #
import numpy as np
import cupy as cp


x_on_gpu0 = cp.array([1,2,3,4,5])
#cp.cuda.Device(1).use()
x_on_gpu1 = cp.array([1,2,3,4,5])
a = cp.cuda.runtime.getDeviceProperties(0)
print(a)
