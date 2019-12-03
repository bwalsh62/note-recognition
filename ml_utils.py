# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:01:32 2019

@author: benja
"""

#%% Import libraries

import numpy as np

#%%

def music_feat_extract(data, fs, music_dict):
    
    # Find index corresponding to each note of interest
    # For now just take note of C, D and E
    
    tp_count = len(data[0,:])
    vals = np.arange(int(tp_count)/2)
    t_period = tp_count/fs
    freqs = vals/t_period
    
    feat_notes = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3']
    center_indices = []
    
    for feat_note in feat_notes:
        center_indices.append(abs(freqs-music_dict[feat_note]).argmin())
    
    # Window width to extract power around center frequencies
    window_width = 3
    
    # Extract features
    
    features = np.empty((data.shape[0],len(feat_notes)))
    for idx, sample in enumerate(data):
        # Integrate over band for each note
        ftransform = np.fft.fft(sample)/len(sample)
        ftransform = ftransform[range(int(len(sample)/2))]
        
        for feat_idx,center_idx in enumerate(center_indices):
            features[idx,feat_idx] = abs(ftransform)[center_idx-window_width:center_idx+window_width].sum()
     
        # Print progress
        if idx % 100 == 0:
            print('Sample {}/{}'.format(idx,data.shape[0]))
        
    return features