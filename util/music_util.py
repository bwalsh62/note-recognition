# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:41:02 2020

@author: Ben Walsh
For Liloquy

TO DO
- Add optional time shifts to add_noise function

(C) 2021 Ben Walsh <ben@liloquy.io>

"""

#%% Import libraries

from numpy.random import normal

#%% Frequency dictionary
#   Dictionary of frequencies (Hz) for musical notes

note_to_freq = {
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
        'C#4': 277.187,
        'D4': 293.66,
        'D#4': 311.13,
        'E4': 329.63,
        'F4': 349.23,
        'F#4': 370.00,
        'G4': 392.00,
        'G#4': 415.312,
        'A4': 440.00,
        'A#4': 466.172,
        'B4': 493.88,
        'C5': 523.25,
        'D5': 587.33,
        'E5': 659.25,
        'F5': 698.46,
        'G5': 783.99
}

#%% add_noise function
# Usage: noisy_array1 = add_noise(array1)
# Useful for synthesizing additional data while training model

def add_noise(in_array, ampl=0.5):
    out_array = in_array + ampl*normal(0,1,len(in_array))
    return out_array

