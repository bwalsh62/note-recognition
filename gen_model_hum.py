# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound
Training on recorded hummed notes

Created on Sun Nov  3 18:12:29 2019
Last updated December 3 2019

@author: Ben Walsh
for liloquy

TO DO
- Try a more advanced/robust algorithm
- Expand features / notes used: 
    B3
    sharps
    another octave
- Efficient windowing/FFT

"""

#%% Import libraries

print('Importing libraries...')

import matplotlib.pyplot as plt
import numpy as np

# ML with sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Pickle to save model
import pickle

# Scipy to read wav files
from scipy.io import wavfile as wav

# Custom library for adding noise
from noise_util import add_noise

# Custom library with tone information
# (Shouldn't have to be in make_tone)
from make_tone import music_dict
from ml_utils import music_feat_extract

#%% Load hummed notes

print('Loading recorded notes for training...')

t_len = 1 # seconds
notes = ['C4','D4','E4','F4','G4','A4']#,'B4']

c_hum = r".\sound_files\Hum_C4.wav"
fs_in_C, C_sig_in = wav.read(c_hum)

d_hum = r".\sound_files\Hum_D4.wav"
fs_in_D, D_sig_in = wav.read(d_hum)

e_hum = r".\sound_files\Hum_E4.wav"
fs_in_E, E_sig_in = wav.read(e_hum)

f_hum = r".\sound_files\Hum_F4.wav"
fs_in_F, F_sig_in = wav.read(f_hum)

g_hum = r".\sound_files\Hum_G4.wav"
fs_in_G, G_sig_in = wav.read(g_hum)

a_hum = r".\sound_files\Hum_A4.wav"
fs_in_A, A_sig_in = wav.read(a_hum)

# Number of notes
n_class = len(notes)
hum_len = 130000
hums = np.empty((n_class,hum_len))
hums[0,:] = C_sig_in[:hum_len,1]
hums[1,:] = D_sig_in[:hum_len,1]
hums[2,:] = E_sig_in[:hum_len,1]
hums[3,:] = F_sig_in[:hum_len,1]
hums[4,:] = G_sig_in[:hum_len,1]
hums[5,:] = A_sig_in[:hum_len,1]

#%% Plot waveform

# Create array of time samples
t_array_len = hums[0,:].shape[0]
t_array = np.arange(t_array_len)

# Plot waveform over short time period to see sine
plt.subplot(121)
plt.plot(t_array/fs_in_C,hums[0,:])
plt.xlim(0.2 ,0.8)
plt.xlabel('Time (s)')
plt.title('Hummed C note ')#at ' + str(tones[0].f0) + ' hz')

plt.subplot(122)
plt.plot(t_array/fs_in_C,hums[0,:])
plt.xlim(0.5 ,0.55)
plt.xlabel('Time (s)')
plt.title('Hummed C note ')#at ' + str(tones[0].f0) + ' hz')

#%% Plot FFT

ftransform = np.fft.fft(hums[4,:])/len(hums[0,:])
ftransform = ftransform[range(int(len(hums[0,:])/2))]
tp_count = len(hums[0,:])
vals = np.arange(int(tp_count)/2)
t_period = tp_count/fs_in_C
freqs = vals/t_period

plt.plot(freqs,abs(ftransform))
plt.xlim((0,500))
plt.show()

#%% Create dataset by adding random shifts and noise

print('Expanding dataset by adding random shifts and noise...')

# Declare number of entries
n_entries = 100

# Initialize matrix where each row contains a noisy sample
X = np.empty((n_entries*n_class,hum_len))

# Add white noise to each element
for idx in range(n_class):
    for sample in range(n_entries):
        X[sample+n_entries*(idx),:] = add_noise(hums[idx,:])
    
#%% Plot example with added noise
plt.plot(t_array/fs_in_C,X[0,:])
plt.show
plt.xlim(0.5 ,0.55)
plt.xlabel('Time (s)')
plt.title('Hummed C note with noise')

#%% Create features by taking FFT and extracting power at each note

print('Extracting FFT features...')

X_feat = music_feat_extract(X,fs_in_C,music_dict)
  
#%% Create training dataset

# Training truth labels
y = []

for idx, note in enumerate(notes):
    for idx in range(n_entries):
        y.append(note)

#%% Implement ML model

print('Training model...')

# Separate training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, random_state=1)

# Implement Decision Tree Model
# Note: With expanded features, max_features and max_leaf_nodes had to be set to
#  length of feature space
# Performance still was low until max_depth was increased
model = tree.DecisionTreeClassifier(max_depth=5,
            max_features=len(notes), max_leaf_nodes=len(notes), random_state=1)
# Fit model to training set
model.fit(X_train, y_train)

# Predict on test set
y_predict = model.predict(X_test)

# See accuracy on test set
print("Accuracy on test set "+str(100*accuracy_score(y_test, y_predict)))

#%% Save model for later

# save the model to disk
modelName = './model.sav'
pickle.dump(model, open(modelName, 'wb'))
 