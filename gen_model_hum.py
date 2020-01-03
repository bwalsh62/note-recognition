# -*- coding: utf-8 -*-
"""

Train ML to recognize notes from input sound
Training on recorded hummed notes

Created on Sun Nov  3 18:12:29 2019
Last updated December 3 2019

@author: Ben Walsh
for liloquy

TO DO
- Expand features / notes used: 
    B3
    sharps
  [ ]  another octave
- Efficient windowing/FFT
[ ] Enhance feature extraction so C3/C4 works 
    - feat_notes = ('C3', 'D3', 'E3', 'F3', 'G3', 'A3') - #this works
    - feat_notes = ('C4', 'D4', 'E4', 'F4', 'G4', 'A4') - #this should work

"""

#%% Import libraries

print('Importing libraries...')

import matplotlib.pyplot as plt
import numpy as np

# ML with sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# SVMs for ML algorithm
from sklearn import svm

# Pickle to save model
import pickle

# Scipy to read wav files
from scipy.io import wavfile as wav

# Custom library for adding noise
from noise_util import add_noise

# Custom library with tone information
# (Shouldn't have to be in make_tone)
# Use new music_dict from liloquy-git... separate from piano part?
from make_tone import freq_dict

# Custom ML library to extract music features and use hum_signals class
from ml_utils import music_feat_extract, hum_signals

#%% Load hummed notes

print('Loading recorded notes for training...')

t_len = 1 # seconds

#notes = ('D3','E3','F3','G3','A3','C4','D4','E4','F4','G4','A4')
notes = ('C4', 'D4', 'E4', 'F4', 'G4', 'A4')

# Define hum_training object based on input list of notes
hum_training = hum_signals(notes)

# Number of notes
n_class = len(hum_training.hums.keys())

#c_hum = r".\sound_files\Hum_C4.wav"
#fs_in_C, C_sig_in = wav.read(c_hum)

# Enforce consistent length of inputs. Should be integrated with t_len
hum_len = 130000
hums = np.empty((n_class,hum_len))

# Initialize truth labels
y=[]

# Declare number of entries to synthesize additional samples
n_entries = 10

# Build truth labels and create hums matrix
for idx, note in enumerate(hum_training.hums.keys()):
    hums[idx,:] = hum_training.hums[note].signal[:hum_len,1]
    for idx in range(n_entries):
        y.append(note)

#%% Plot waveform to illustrate an example
test_note = 'C4'
fs = hum_training.hums[test_note].fs

# Create array of time samples
t_array_len = hums[notes.index(test_note),:].shape[0]
t_array = np.arange(t_array_len)

# Plot waveform over short time period to see sine
plt.subplot(121)
plt.plot(t_array/fs,hums[notes.index(test_note),:])
plt.xlim(0.2 ,0.8)
plt.xlabel('Time (s)')
plt.title('Hummed {}: {} hz'.format(test_note,freq_dict[test_note]))

plt.subplot(122)
plt.plot(t_array/fs,hums[notes.index(test_note),:])
plt.xlim(0.5 ,0.55)
plt.xlabel('Time (s)')
plt.title('Hummed {}: {} hz'.format(test_note,freq_dict[test_note]))

#%% Plot FFT

ftransform = np.fft.fft(hums[0,:])/len(hums[0,:])
ftransform = ftransform[range(int(len(hums[0,:])/2))]
tp_count = len(hums[0,:])
vals = np.arange(int(tp_count)/2)
t_period = tp_count/fs
freqs = vals/t_period

plt.plot(freqs,abs(ftransform))
plt.xlim((0,500))
plt.show()

#%% Create dataset by adding random shifts and noise

print('Expanding dataset by adding random shifts and noise...')

# Initialize matrix where each row contains a noisy sample
X = np.empty((n_entries*n_class,hum_len))

# Add white noise to each element
for idx in range(n_class):
    for sample in range(n_entries):
        X[sample+n_entries*(idx),:] = add_noise(hums[idx,:])
    
#%% Plot example with added noise
plt.plot(t_array/fs,X[0,:])
plt.show
plt.xlim(0.5 ,0.55)
plt.xlabel('Time (s)')
plt.title('Hummed C note with noise')

#%% Create features by taking FFT and extracting power at each note

print('Extracting FFT features...')

# Ideally this should be equal to 'notes'
feat_notes = ('C3', 'D3', 'E3', 'F3', 'G3', 'A3')
#feat_notes = ('C4', 'D4', 'E4', 'F4', 'G4', 'A4')

X_feat = music_feat_extract(X,fs,freq_dict,feat_notes)

#%% Implement ML model

print('Training model...')

# Separate training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, random_state=1)

# Implement Decision Tree Model
# Note: With expanded features, max_features and max_leaf_nodes had to be set to
#  length of feature space
# Performance still was low until max_depth was increased
#model = tree.DecisionTreeClassifier(max_depth=5,
#            max_features=len(notes), max_leaf_nodes=len(notes), random_state=1)
# Note: Decision Tree was not generalizing to new recording
# After feature exploration, found that it was overfitting to an incorrect feature extraction
model = svm.SVC()
# Fit model to training set
model.fit(X_train, y_train)

# Predict on test set
y_predict = model.predict(X_test)

# See accuracy on test set
print("Accuracy on test set "+str(100*accuracy_score(y_test, y_predict)))

#%% Save model for later

modelName = './model.sav'
pickle.dump(model, open(modelName, 'wb'))
 