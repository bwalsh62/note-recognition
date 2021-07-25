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
import numpy as np
from numpy.random import normal
import os
import sys

import wave

# Add custom modules to path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from util.ml_util import feat_extract

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
HUM_FPATH = r"C:\Users\benja\OneDrive\Documents\Python\liloquy-git\note-recognition\sound_files"

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

# Initialize dictionary to look up wav file paths for each note
lib_note_path = {}

for lib_note in LIB_NOTES:

    # Build dictionary
    lib_note_path[lib_note] = {}
    lib_note_path[lib_note]['piano'] = os.path.join(MUSIC_FPATH,"Piano_{}_2p4s.wav".format(lib_note))
    lib_note_path[lib_note]['hum'] = os.path.join(HUM_FPATH,"Hum_{}.wav".format(lib_note))

    note_to_sound[lib_note] = {}
    
    if os.path.exists(lib_note_path[lib_note]['piano']):
        note_to_sound[lib_note]['piano'] = mixer.Sound(lib_note_path[lib_note]['piano'])
    else:
        print("{} does not exist".format(lib_note_path[lib_note]['piano']))
    
    if os.path.exists(lib_note_path[lib_note]['hum']):
        note_to_sound[lib_note]['hum'] = mixer.Sound(lib_note_path[lib_note]['hum'])
    else:
        print("{} does not exist".format(lib_note_path[lib_note]['hum']))

#%% add_noise function
# Usage: noisy_array1 = add_noise(array1)
# Useful for synthesizing additional data while training model

def add_noise(in_array, ampl=0.5):
    out_array = in_array + ampl*normal(0,1,len(in_array))
    return out_array

#%% melody_transcribe function
# Usage: predicted_notes = melody_transcribe(melody, model)
# Outputs array of notes e.g. ['C4', 'E4'] based on 
# input melody and predictive model
# 
# Example:
# fs, wav_signal = wav.read(hum_wav_file)
# predicted_notes = melody_transcribe(melody=wav_signal, fs, model, note_len, SCALE)

def melody_transcribe(melody, fs, model, note_len, scale, debug=False):
    
    # Estimate note_total by rounding the input note_len to length of input melody
    note_total = np.int(round(melody.shape[0]/note_len))
    if debug:
        print('note_total = {}'.format(note_total))
        print('melody.shape = {}'.format(melody.shape))
        print('note_len = {}'.format(note_len))
    
    # Pad melody array so length is consistent
    melody_clean = np.zeros((note_len*note_total,))
    # Take single dimension to simplify dual-channel recording
    melody_single_ch = melody[:,1]
    
    if len(melody_single_ch)<=len(melody_clean):
        melody_clean[:len(melody)] = melody_single_ch
    else:
        if debug:
            print('len(melody_single_ch) = {}'.format(len(melody_single_ch)))
        # Truncate melody if longer than expected
        note_samp_to_drop = np.int((len(melody_single_ch) - len(melody_clean))/note_total)
        if debug:
            print('note_samp_to_drop = {}'.format(note_samp_to_drop))
        note_input_len = np.int(round(len(melody_single_ch)/note_total))
        for note_idx in range(note_total):
            try:
                melody_clean[note_len*note_idx:note_len*(note_idx+1)] = \
                    melody_single_ch[note_input_len*note_idx:note_input_len*(note_idx+1)-note_samp_to_drop]
            except:
                # Account for odd rounding
                melody_clean[note_len*note_idx:note_len*(note_idx+1)] = \
                    melody_single_ch[note_input_len*note_idx:note_input_len*(note_idx+1)-note_samp_to_drop-1]
    
    # Initialize matrix of notes
    notes = np.empty((note_total, note_len))
    for note_idx in range(note_total):
        notes[note_idx,:] = melody_clean[note_len*note_idx:note_len*(note_idx+1)]

    X_feat = feat_extract(notes, fs, note_to_freq, scale)
    
    predicted_notes = model.predict(X_feat)
    
    return predicted_notes

#%% wav_file_append function
# Usage: wav_file_concat = wav_file_concat(wav_file1, wav_file2, merge_name='./concat.wav')

def wav_concat(wav_file1, wav_file2, merge_name='./concat.wav'):
    
    # Error checking to wav_file_in existence
    if not(os.path.exists(wav_file1)):
        print('Input file does not exist: {}'.format(wav_file1))
    if not(os.path.exists(wav_file2)):
        print('Input file does not exist: {}'.format(wav_file2))
    
    # Read first wav file
    obj = wave.open(wav_file1,'r')
    n_frames1 = obj.getnframes()
    wav1_data = obj.readframes(n_frames1)
    n_channels = obj.getnchannels()
    samp_width = obj.getsampwidth()
    fs = obj.getframerate()
    obj.close()

    # Read second wav file
    obj = wave.open(wav_file2,'r')
    n_frames2 = obj.getnframes()
    wav2_data = obj.readframes(n_frames2)
    obj.close()
    
    # Write new wav file
    obj = wave.open(merge_name,'wb')
    obj.setnframes(n_frames1+n_frames2)
    obj.setnchannels(n_channels)
    obj.setsampwidth(samp_width)
    obj.setframerate(fs)
    obj.writeframes(wav1_data+wav2_data) 
    obj.close()
        
    return merge_name

#%% Append individual note wav files into a single wav file according to the
#     input notes
def melody_write(notes, instr='piano', fname='melody.wav'):
    
    # hum_wav_file = fr"C:\Users\benja\OneDrive\Documents\Python\liloquy-git\note-recognition\sound_files\Hum_{note}.wav"
    # fs_in, wav_sig_in = wav.read(hum_wav_file)
    
    for idx, note in enumerate(notes):
        if idx==0:
            # Initialize wav_file with first note
            melody_wav = lib_note_path[note][instr]
        elif idx>0:
            # Append next note with original wav file
            melody_wav = wav_concat(melody_wav, lib_note_path[note][instr], fname)
    
    return mixer.Sound(melody_wav)

#%% Class to define note to playback with associated metadata

class Note:
    
    fs = 44100  # Sampling frequency in Hz
    
    def __init__(self, note, instr='piano'):        
        #self.f0 = f0 # frequency in Hz
        self.note = note # Example 'C4'
        self.f0 = note_to_freq[note]
        self.instr = instr
        self.sound = note_to_sound[note][instr]

#%% Class to define notes to playback with associated metadata

class Melody:
    
    fs = 44100  # Sampling frequency in Hz
    
    def __init__(self, notes, instr='piano'):        
        #self.f0 = f0 # frequency in Hz
        self.notes = notes # Example ['C4', 'E4']
        self.freqs = [note_to_freq[note] for note in notes]
        self.instr = instr
        self.sound = melody_write[notes]
        