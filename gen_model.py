# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound

Created on Sun Nov  3 18:12:29 2019
Last updated November 7 2019

@author: Ben Walsh

TO DO
- Extract FFT features
- Try a more advanced algorithm

"""

#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np

# ML with sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Custom library to make tones
from make_tone import tone, music_dict
from noise_util import add_noise

#%% Generate pure sine waves

t_len = 0.1 # seconds
notes = ['C4','D4','E4','F4','G4','A4','B4']
freqs = [music_dict[note] for note in notes]
tones = [tone(f,t_len) for f in freqs]

#%% Plot waveform

# Create array of time samples
t_array_len = tones[0].time_len*tones[0].fs
t_array = np.arange(t_array_len)

# Plot waveform over short time period to see sine
plt.subplot(121)
plt.plot(t_array/tones[0].fs,tones[0].sin_wave)
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tones[0].f0) + ' hz')

# Plot waveform over short time period to see sine
plt.subplot(122)
plt.plot(t_array/tones[1].fs,tones[1].sin_wave)
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tones[1].f0) + ' hz')

#%% Create dataset by adding random shifts and noise

# Declare number of entries
n_entries = 200
n_class = len(tones)

# Initialize matrix where each row contains a sine wave
X = np.empty((n_entries*n_class,len(tones[0].sin_wave)))

# Add white noise to each element
for idx, tone_class in enumerate(tones):
    for sample in range(n_entries):
        X[sample+n_entries*(idx),:] = add_noise(tone_class.sin_wave)
    
#%% Plot example with added noise
plt.plot(t_array/tones[0].fs,X[0,:])
plt.show
plt.xlim(0 ,0.05)
plt.xlabel('Time (s)')
plt.title('Sine wave at ' + str(tones[0].f0) + ' hz with noise')

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
X = X[:,:200]

# Training truth labels
y = np.empty((n_entries*n_class,1))

for idx, note in enumerate(notes):
    y[n_entries*(idx):n_entries*(idx+1)]=str(idx)

#%% Implement ML model

# Separate training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Implement Decision Tree Model
model = tree.DecisionTreeClassifier(max_depth=10,
            max_features=10, max_leaf_nodes=10, random_state=1)
# Fit model to training set
model.fit(X_train, y_train)

# Predict on test set
y_predict = model.predict(X_test)

# See accuracy on test set
print("Accuracy on test set "+str(100*accuracy_score(y_test, y_predict)))