# -*- coding: utf-8 -*-
"""
General definitions through utility files

(C) 2021 Ben Walsh <ben@liloquy.io>

"""

#%% Import libraries

import os

#%%

AUDIO_FOLDER = r"../1_audio"
DATA_FOLDER = r"../3_data"
MODEL_FOLDER = r"../4_model"

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    