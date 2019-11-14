# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:29:21 2019

Tests pre-trained model and plays predictions in piano
To generate new model, run gen_model.py

Last updated November 13, 2019

@author: Ben Walsh
for liloquy

TO DO
- Test against hummed note
- Investigate if original piano was overwritten to play 2x

"""
#%% Import libraries

# Pickle to load pre-trained model
import pickle

# Numpy for matrices and arrays
import numpy as np

# Custom library to make tones
from make_tone import tone, music_dict

# Custom library to play piano notes
from piano_notes import note_class

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
X[0] = tones[4].sin_wave 
X[1] = tones[1].sin_wave
X = X[:,:200]
 
#%% Test on 2 tones    
predicted_notes = loaded_model.predict(X)
for note in predicted_notes:
    print("Predicted note: "+note)

#%% Play piano notes at predicted notes

note_predict = note_class(note=predicted_notes[0])
note_predict.sound.play(1)
