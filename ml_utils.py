# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:01:32 2019

@author: benja

Music feature extraction used by gen_model_hum.py to train ML model

"""

#%% Import libraries

import numpy as np

#%%

def music_feat_extract(data, fs, freq_dict):
    
    # Find index corresponding to each note of interest
    # For now just take note of C, D and E
    
    tp_count = len(data[0,:])
    vals = np.arange(int(tp_count)/2)
    t_period = tp_count/fs
    freqs = vals/t_period
    
    feat_notes = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3']
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
    return features