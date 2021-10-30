# -*- coding: utf-8 -*-
"""
Music feature extraction used by gen_model_hum.py to train ML model
... and other utility functions

TO DO
- Generalize path / add PATH variable for training_data files
- load_training_data does not need both train_len and t_len

(C) 2021 Ben Walsh <ben@liloquy.io>

"""

#%% Import libraries

import numpy as np
import pandas as pd
from scipy.io import wavfile as wav
import os
import sys

# Add custom modules to path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from util import AUDIO_FOLDER, DATA_FOLDER, MODEL_FOLDER

#%% Define constants

train_type = 'hum' # hard-coded for now, may grow in the future

training_data = {
        'D3': os.path.join(AUDIO_FOLDER, train_type, "{}_D3.wav".format(train_type)),
        'E3': os.path.join(AUDIO_FOLDER, train_type, "{}_E3.wav".format(train_type)),
        'F3': os.path.join(AUDIO_FOLDER, train_type, "{}_F3.wav".format(train_type)),
        'G3': os.path.join(AUDIO_FOLDER, train_type, "{}_G3.wav".format(train_type)),
        'A3': os.path.join(AUDIO_FOLDER, train_type, "{}_A3.wav".format(train_type)),
        'C4': os.path.join(AUDIO_FOLDER, train_type, "{}_C4.wav".format(train_type)),
        'D4': os.path.join(AUDIO_FOLDER, train_type, "{}_D4.wav".format(train_type)),
        'E4': os.path.join(AUDIO_FOLDER, train_type, "{}_E4.wav".format(train_type)),
        'F4': os.path.join(AUDIO_FOLDER, train_type, "{}_F4.wav".format(train_type)),
        'G4': os.path.join(AUDIO_FOLDER, train_type, "{}_G4.wav".format(train_type)),
        'A4': os.path.join(AUDIO_FOLDER, train_type, "{}_A4.wav".format(train_type))
}

#%%
# Usage: X_train = feat_extract(data, fs, freq_dict, feat_notes)
# Output X_train is a DataFrame with feat_notes as columns

def feat_extract(data, fs, freq_dict, feat_notes):
    
    # Find index corresponding to each note of interest
    # For now just take note of C, D and E
    
    tp_count = len(data[0,:])
    vals = np.arange(int(tp_count)/2)
    t_period = tp_count/fs
    freqs = vals/t_period
    
    center_indices = []
    
    for feat_note in feat_notes:
        center_indices.append(abs(freqs-freq_dict[feat_note]).argmin())
    
    # Window width to extract power around center frequencies
    window_width = 2
    
    # Extract features
    
    features = np.empty((data.shape[0],len(feat_notes)))
    for idx, sample in enumerate(data):
        # Integrate over band for each note
        ftransform = np.fft.fft(sample)/len(sample)
        ftransform = ftransform[range(int(len(sample)/2))]
        
        for feat_idx,center_idx in enumerate(center_indices):
            features[idx,feat_idx] = abs(ftransform)[center_idx-window_width:center_idx+window_width].sum()
     
        # Normalize features in each sample
        features[idx,:] = features[idx,:]/features[idx,:].sum()
        
        # Print progress
        if (idx+1) % 100 == 0:
            print('Feat extract: sample {}/{}'.format(idx+1,data.shape[0]))
            
    # Return features   
    return pd.DataFrame(features, columns=feat_notes)

#%% Training class

class signal:
    
    def __init__(self, note='C4'):        
        self.note = note 
        fs, signal = wav.read(training_data[note])
        self.fs = fs
        self.signal = signal
        self.wav_file = training_data[note]
    
class signals(signal):
    
    def __init__(self, notes=('C4','D4','E4','F4','G4','A4')):        
        self.signals = dict()
        for note in notes:
            self.signals[note] = signal(note)
        self.notes = len(set(notes))
            
#%%

def load_training_data(notes = ('C4', 'D4', 'E4')):

    t_len = 2.5 # seconds
    
    # Define hum_training object based on input list of notes
    training_data = signals(notes)
    
    # Assume constant fs and take the first note's fs
    fs = training_data.signals[notes[0]].fs
    
    # Number of notes
    n_class = training_data.notes
    
    # Enforce consistent length of inputs. Should be integrated with t_len
    train_len = int(t_len*fs) #130000 # in ms
    X = np.empty((n_class, train_len))
    
    # Initialize truth labels
    y=[]
    
    # Build truth labels and create hums matrix
    for idx, note in enumerate(notes): 
        X[idx,:] = training_data.signals[note].signal[:train_len,1]
        y.append(note)
    
    return X, y, fs
    