# -*- coding: utf-8 -*-
"""
Class to instantiate a tone, with a default
length of time = 1 second, sampling
frequency of 44.1kHz and an input
fundamental frequency

# Usage:
    from make_tone import tone
    f0 = 440
    tone1 = tone(f0)

Created on Tue Aug 14 22:22:37 2018
Last updated: September 5, 2018

Author: Ben Walsh
"""

#%% Import standard libraries

import numpy as np

#%% Tone class

class tone:
    
    fs = 44100  # Sampling frequency in Hz
    
    def __init__(self, f0, time_len=1):        
        self.f0 = f0 # frequency in Hz
        self.time_len = time_len # time in seconds
        t_array = np.arange(self.time_len*self.fs)
        self.sin_wave = 5*np.sin(2*np.pi*self.f0*t_array/self.fs)
    
    def changeLength(self, new_time_len):
        self.time_len = new_time_len
        t_array = np.arange(self.time_len*self.fs)
        self.sin_wave = 5*np.sin(2*np.pi*self.f0*t_array/self.fs)
    