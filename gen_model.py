# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound

Created on Sun Nov  3 18:12:29 2019
Last updated November 5 2019

@author: Ben Walsh

TO DO
- Extract FFT features
- Try 4 notes/classes
- import music_dict from another file

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

mus_dict = {
        'D2': 73.42,
        'E2': 82.41,
        'F2': 87.31,
        'G2': 98.00,
        'A2': 110.00,
        'B2': 123.47,
        'C3': 130.81,
        'D3': 146.83,
        'E3': 164.81,
        'F3': 174.61,
        'G3': 196.00,
        'A3': 220.00,
        'B3': 246.94,
        'C4': 261.63,
        'D4': 293.66,
        'E4': 329.63,
        'F4': 349.23,
        'G4': 392.00,
        'A4': 440.00,
        'B4': 493.88,
        'C5': 523.25,
        'D5': 587.33,
        'E5': 659.25,
        'F5': 698.46,
        'G5': 783.99
}

t_len = 0.1 # seconds
notes = ['C4','E4','G4']
f1 = mus_dict[notes[0]] 
tone1 = tone(f1,t_len)

f2 = mus_dict[notes[1]]
tone2 = tone(f2,t_len)

f3 = mus_dict[notes[2]]
tone3 = tone(f3,t_len)

tone_classes = [tone1.sin_wave, tone2.sin_wave, tone3.sin_wave]

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
n_class = len(tone_classes)

# Initialize matrix where each row contains a sine wave
X = np.empty((n_entries*n_class,len(tone1.sin_wave)))

# Add white noise to each element
for idx, tone_class in enumerate(tone_classes):
    for sample in range(n_entries):
        X[sample+n_entries*(idx),:] = add_noise(tone_class)
    
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

#for idx, note in enumerate(notes):
#    y[n_entries*(idx):n_entries*(idx+1)]=note

y[:n_entries] = '1'#f1
y[n_entries:n_entries*2] = '2'#f2
y[n_entries*2:n_entries*3] = '3'#f3


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