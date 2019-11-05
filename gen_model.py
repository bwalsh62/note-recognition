# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound

Created on Sun Nov  3 18:12:29 2019

@author: Ben Walsh

TO DO
- Extract FFT features
- Try 3 notes/classes

"""

#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np

# ML with sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

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
plt.subplot(121)
plt.plot(t_array/tone1.fs,tone1.sin_wave)
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone1.f0) + ' hz')

# Plot waveform over short time period to see sine
plt.subplot(122)
plt.plot(t_array/tone2.fs,tone2.sin_wave)
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone2.f0) + ' hz')

#%% Create dataset by adding random shifts and noise

# Declare number of entries
n_entries = 100
n_class = 2

# Initialize matrix where each row contains a sine wave
X = np.empty((n_entries*n_class,len(tone1.sin_wave)))

# Add white noise to each element
for sample in range(n_entries):
    X[sample,:] = add_noise(tone1.sin_wave)

for sample in range(n_entries):
    X[sample+n_entries,:] = add_noise(tone2.sin_wave)
    
#%% Plot example with added noise
plt.plot(t_array/tone1.fs,X[0,:])
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tone1.f0) + ' hz with noise')

##%% Create features by taking FFT and extracting power at each note
## Skip for now
#
## Take FFT of sine wave
#sp1 = np.fft.fft(tone1.sin_wave)
#sp2 = np.fft.fft(tone2.sin_wave)
#freqs = np.fft.fftfreq(len(tone1.sin_wave))
#freqsHz = abs(freqs*tone1.f0)
#
## Plot FFT for testing
#plt.subplot(121)
#plt.plot(freqsHz, sp1.real)
#plt.subplot(122)
#plt.plot(freqsHz, sp2.real)
#plt.show
#
## Integrate over band for each note

#%% Create training dataset

# For simplicity for now just take first 100 samples as input features
X = X[:,:100]

# Training truth labels
y = np.empty((n_entries*n_class,1))
y[:n_entries] = '200'
y[n_entries:n_entries*2] = '282'

#%% Implement ML model

# Separate training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Implement Decision Tree Model
model = tree.DecisionTreeClassifier(max_depth=3,
            max_features=3, max_leaf_nodes=3, random_state=1)
# Fit model to training set
model.fit(X_train, y_train)

# Predict on test set
y_predict = model.predict(X_test)

# See accuracy on test set
accuracy_score(y_test, y_predict)