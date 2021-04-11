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

from pygame import mixer
from numpy.random import normal
import os

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

#%% Define saved notes

mixer.init()

MUSIC_FPATH = r"C:\Users\benja\OneDrive\Documents\Python\liloquy-git\piano-gui\music_files\piano"

note_to_sound = {}
LIB_NOTES = ('C4',
             'Db4', 
             'D4', 
             'Eb4', 
             'E4', 
             'F4', 
             'Gb4', 
             'G4', 
             'Ab4',
             'A4', 
             'Bb4',
             'B4')

for lib_note in LIB_NOTES:
    lib_note_path = os.path.join(MUSIC_FPATH,"Piano_{}_2p4s.wav".format(lib_note))
    if os.path.exists(lib_note_path):
        note_to_sound[lib_note] = mixer.Sound(lib_note_path)
    else:
        print("{} does not exist".format(lib_note_path))

#%% add_noise function
# Usage: noisy_array1 = add_noise(array1)
# Useful for synthesizing additional data while training model

def add_noise(in_array, ampl=0.5):
    out_array = in_array + ampl*normal(0,1,len(in_array))
    return out_array

#%% Class to define notes to playback with associated metadata

class Note:
    
    fs = 44100  # Sampling frequency in Hz
    
    def __init__(self, note, instr='piano'):        
        #self.f0 = f0 # frequency in Hz
        self.note = note # Example C4
        self.f0 = note_to_freq[note]
        self.instr = instr
        self.sound = note_to_sound[note]
        