import os
import pandas as pd
import numpy as np
import utils
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import re


def augment_song(rate, data):
    """
    Performs data augmentation. DATA is a numpy array with 2 channels with sample rate RATE.
    Returns a list of augmentations, each of which is a numpy array with the same
    shape and data type as DATA.
    """
    
    augmentations = []
    augmentations.append(data)
    
    swap = np.ndarray(data.shape, dtype=data.dtype)
    swap[:,0] = data[:,1]
    swap[:,1] = data[:,0]
    augmentations.append(swap)
    
    return augmentations



def parse_irmas_trainset (source, dest):
    """
    Walks recursively at SOURCE directory (relative to the current directory) to crawl all
    .wav files. Copies all files from SOURCE to DEST. Additionally, this function performs
    data augmentation on each file using the function AUGMENT_SONG(). If DEST does not exist
    then a new directory named DEST is created. Both SOURCE and DEST are relative to the
    current directory.
    
    Example Usage: irmasTestUtils.parse_irmas_trainset("../IRMAS-TrainingData", "../Preprocessed_Trainset")
    """
    
    files_list = []
    
    # Load the source datapath
    dataPath = os.path.abspath(source)
    
    # Create the new directory if it doesn't exist
    if not os.path.isdir(os.path.abspath(dest)):
        os.makedirs(os.path.abspath(dest))
    
    # Crawl starting at SOURCE recursively
    r = 0
    for root, dir, files in os.walk(dataPath, topdown = True):
        r += 1
        print("Processing directory: " + str(r))
        for file in files:

            # Examine only .wav files
            if file[-4:] != ".wav":
                continue
            
            # Determines the type of instrument from the file name
            match = re.search(utils.INST_PATTERN, file)
            if not match:
                continue
            
            rate, data = read_wav (root + '/' + file)
            category = match.group(0)
            
            # Perform data augmentation (e.g. flipping the channels)
            augmentations = augment_song(rate, data)
            
            # Write the song to DEST
            count = 0
            for augmentation in augmentations:
                write_wav (dest + '/' + file[:-4] + '[' + str(count) + '].wav', rate, augmentation)
                count += 1
    
    return
    
