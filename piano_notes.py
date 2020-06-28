# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:56:26 2019

@author: Ben Walsh
for liloquy

Music from: http://theremin.music.uiowa.edu/MISpiano.html

Last updated: November 13, 2019

# TO DO
# - Expand sound_dict beyone one octave

"""

#%% Import libraries
from pygame import mixer
import os
import sys

# Add custom modules to path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from util.music_util import freq_dict

#%%
mixer.init()

# Paths to piano wav files
music_fpath = r"..\..\piano-gui\music_files\piano"
C_path = os.path.join(music_fpath,"Piano.mf.C4_2p4s.wav")
Csharp_path = os.path.join(music_fpath,"Piano.mf.Db4_2p5s.wav")
D_path = os.path.join(music_fpath,"Piano.mf.D4_2p4s.wav")
Dsharp_path = os.path.join(music_fpath,"Piano.mf.Eb4_2p5s.wav")
E_path = os.path.join(music_fpath,"Piano.mf.E4_2p4s.wav")
F_path = os.path.join(music_fpath,"Piano.mf.F4_2p4s.wav")
Fsharp_path = os.path.join(music_fpath,"Piano.mf.Gb4_2p5s.wav")
G_path = os.path.join(music_fpath,"Piano.mf.G4_2p4s.wav")
Gsharp_path = os.path.join(music_fpath,"Piano.mf.Ab4_2p5s.wav")
A_path = os.path.join(music_fpath,"Piano.mf.A4_2p4s.wav")
Asharp_path = os.path.join(music_fpath,"Piano.mf.Bb4_2p5s.wav")
B_path = os.path.join(music_fpath,"Piano.mf.B4_2p4s.wav")

# Define sounds
sound_C = mixer.Sound(C_path)
sound_Csharp = mixer.Sound(Csharp_path)
sound_D = mixer.Sound(D_path)
sound_Dsharp = mixer.Sound(Dsharp_path)
sound_E = mixer.Sound(E_path)
sound_F = mixer.Sound(F_path)
sound_Fsharp = mixer.Sound(Fsharp_path)
sound_G = mixer.Sound(G_path)
sound_Gsharp = mixer.Sound(Gsharp_path)
sound_A = mixer.Sound(A_path)
sound_Asharp = mixer.Sound(Asharp_path)
sound_B = mixer.Sound(B_path)

sound_dict = {
        'C4': sound_C,
        'C#4': sound_Csharp,
        'D4': sound_D,
        'D#4': sound_Dsharp,
        'E4': sound_E,
        'F4': sound_F,
        'F#4': sound_Fsharp,
        'G4': sound_G,
        'G#4': sound_Gsharp,
        'A4': sound_A,
        'A#4': sound_Asharp,
        'B4': sound_B,
}

#%% Try to combine into class

class note_class:
    
    fs = 44100  # Sampling frequency in Hz
    
    def __init__(self, note, instr='piano'):        
        #self.f0 = f0 # frequency in Hz
        self.note = note # Example C4
        self.f0 = freq_dict[note]
        self.instr = instr
        self.sound = sound_dict[note]
