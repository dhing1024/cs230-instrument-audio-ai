
import os
from scipy.io.wavfile import read as read_wav
import pandas as pd
import numpy as np

RATE = 44100
NCHANNELS = 2
NFRAMES = 132299

# Walk through all files (recursively) under pathname
# For each file that is a .wav, copy it's data into
# the list
def load_wav_dataset (pathname):
    files_list = []
    dataPath = os.path.abspath(pathname)
    for root, dir, files in os.walk(dataPath, topdown = True):
        for file in files:
            if file [-4:] == ".wav":
                rate, data = read_wav (root + '/' + file)
                dict = {
                    "file": file,
                    "dir": dir,
                    "root": root,
                    "rate": rate,
                    "data": data,
                    "nframes": data.shape[0],
                    "nchannels": data.shape[1]
                }
                files_list.append(dict)
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
              "data": sum,
              "nframes": sum.shape[0],
              "nchannels": sum.shape[1]
            }
            dataset.append(dict)
            
    df = pd.DataFrame (dataset);
    return df