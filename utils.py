
import os
from scipy.io.wavfile import read as read_wav
import pandas as pd
import numpy as np
import re

RATE = 44100
NCHANNELS = 2
NFRAMES = 132299

INST_PATTERN = '(\[)(...)(\])'

INSTRUMENTS = {
    'cel': 0,
    'cla': 1,
    'flu': 2,
    'gac': 3,
    'gel': 4,
    'org': 5,
    'pia': 6,
    'sax': 7,
    'tru': 8,
    'vio': 9,
    'voi': 10,
}

NINSTRUMENTS = 11


def normalize_audio (arr):
    norm = np.linalg.norm(arr)
    return arr / norm


# Walk through all files (recursively) under pathname
# For each file that is a .wav, copy it's data into
# the list. Returns the full dataset, with labels,
# as a Pandas DataFrame
def load_wav_dataset (pathname):
    files_list = []
    dataPath = os.path.abspath(pathname)
    for root, dir, files in os.walk(dataPath, topdown = True):
        for file in files:
            
            # Examine only .wav files
            if file [-4:] == ".wav":
                rate, data = read_wav (root + '/' + file)
                cats = np.zeros(shape=(NINSTRUMENTS, 1), dtype=np.int8)
                
                # Determines the type of instrument from the file name
                match = re.search(INST_PATTERN, file)
                if (match):
                    cats[INSTRUMENTS[match.group(2)]] = 1
                    
                    dict = {
                        "file": file,
                        "dir": dir,
                        "root": root,
                        "rate": rate,
                        "data": normalize_audio (data),
                        "nframes": data.shape[0],
                        "nchannels": data.shape[1],
                        "label": cats,
                        "instruments": match.group(2)
                    }
                    
                    files_list.append(dict)
    
    # Returns the data as a Pandas DataFrame
    df = pd.DataFrame(files_list)
    return df

# Loads the dataframe and generates new combinations of instruments
def combine_dataset (df, num_combinations, combine_size):
    dataset = []
    for i in range (num_combinations):
        sample = df.sample(n = combine_size)
        sum = np.zeros(shape = (NFRAMES, NCHANNELS))
        
        valid = True
        for j in range (combine_size):
            data_j = sample.iloc[j]['data']
            
            if data_j.shape[0] != NFRAMES or data_j.shape[1] != NCHANNELS:
                valid = False
                break
            
            sum = sum + data_j
        
        if valid:
            dict = {
              "rate": RATE,
              "data": normalize_audio (sum),
              "nframes": sum.shape[0],
              "nchannels": sum.shape[1]
            }
            dataset.append(dict)
            
    df = pd.DataFrame (dataset);
    return df
