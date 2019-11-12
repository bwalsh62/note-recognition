# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:29:21 2019

Tests pre-trained model
To generate new model, run gen_model.py

@author: Ben Walsh

TO DO
- Test against hummed note

"""
#%% Import libraries

# Pickle to load pre-trained model
import pickle

# Numpy for matrices and arrays
import numpy as np

# Custom library to make tones
from make_tone import tone, music_dict

#%% Import model
modelName = './model.sav'
 
# load the model from disk
loaded_model = pickle.load(open(modelName, 'rb'))

#%% Generate tone test set

t_len = 0.1 # seconds
notes = ['C4','D4','E4','F4','G4','A4','B4']
freqs = [music_dict[note] for note in notes]
tones = [tone(f,t_len) for f in freqs]

X = np.empty((2,len(tones[0].sin_wave)))
X[0] = tones[0].sin_wave 
X[1] = tones[1].sin_wave
X = X[:,:200]
 
#%% Test on 2 tones    
loaded_model.predict(X)