# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound

Created on Sun Nov  3 18:12:29 2019

@author: Ben Walsh
"""

#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np

# Custom library to make tones
from make_tone import tone
from noise_util import add_noise

#%% Generate pure sine waves

t_len = 0.1 # seconds
f1 = 200 # in Hz
tone1 = tone(f1,t_len)

f2 = 282 # in Hz
tone2 = tone(f2,t_len)

#%% Plot waveform

# Create array of time samples
t_array_len = tone1.time_len*tone1.fs
t_array = np.arange(t_array_len)

# Plot waveform over short time period to see sine
plt.plot(t_array/tone1.fs,tone1.sin_wave)
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone1.f0) + ' hz')

# Plot waveform over short time period to see sine
plt.plot(t_array/tone2.fs,tone2.sin_wave)
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone2.f0) + ' hz')

#%% Create dataset by adding random shifts and noise

# Declare number of entries
n_entries = 100

# Initialize matrix where each row contains a sine wave
X = np.empty((n_entries,len(tone1.sin_wave)))

# Add white noise to each element
for sample in range(n_entries):
    X[sample,:] = add_noise(tone1.sin_wave)
    
#%% Plot example with added noise
plt.plot(t_array/tone1.fs,X[0,:])
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone1.f0) + ' hz with noise')

#%% Create features by taking FFT and extracting power at each note

# Take FFT of sine wave
sp = np.fft.fft(tone1.sin_wave)
freqs = np.fft.fftfreq(t_array.shape[-1])
freqsHz = abs(freqs*tone1.f0)

# Plot FFT for testing
plt.plot(freqsHz, sp.real)
plt.show

# Integrate over band for each note

#%% Implement 