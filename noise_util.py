# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:01:04 2019

@author: benja

Usage: noisy_array1 = add_noise(array1)

"""

from numpy.random import normal

def add_noise(in_array,ampl=0.5):
    out_array = in_array + ampl*normal(0,1,len(in_array))
    return out_array