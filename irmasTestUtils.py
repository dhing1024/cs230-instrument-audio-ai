import os
import numpy as np
import utils
import shutil
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import random

def parse_label_test_label_from_string (label):
    "Converts a string label into a numpy label. Strips '[' and ']' if they exist"
    
    new_label = label.lstrip('[').rstrip(']')
    as_int = [int(char) for char in new_label]
    print(np.array(as_int).reshape(len(new_label), 1))
    
    


def parse_irmas_test_label_from_txt (filename):
    """
    Parses a text file FILENAME and generates a label for the musical instruments
    it contains. Each label is a string of utils.NINSTRUMENTS length, with a 0
    if the text file does not contain the instrument and a 1 if the text file
    does contain the instrument. Each position in the label represents a different
    instrument
    """
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    
    label = ["0"] * utils.NINSTRUMENTS
    
    for line in lines:
        label[utils.INSTRUMENTS[line.strip()]] = "1"
            
    return "".join(label)



def sample_from_audio (rate, data):
    length = data.shape[0]
    start = random.randint(0, length - rate * 3 - 1)
    return data[start: start + rate * 3 - 1,:]



def parse_irmas_testset (source, dest):
    """
    Walks recursively at SOURCE directory (relative to the current directory) to crawl all
    .wav files. Copies all files from SOURCE to DEST. If DEST does not exist
    then a new directory named DEST is created. Both SOURCE and DEST are relative to the
    current directory.
    
    Example Usage: irmasTestUtils.parse_irmas_testset("../IRMAS-TestingData", "../Preprocessed_Testset")
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
                
            rate, data = read_wav (root + '/' + file)
            cut_data = sample_from_audio (rate, data)
                        
            # Parse the labels from the associated .txt file
            base_name = file[:-4]
            label = parse_irmas_test_label_from_txt(root + '/' + base_name + '.txt')
            new_file_name = dest + '/[' + label + '] ' + base_name + '.wav'
            
            write_wav (new_file_name, rate, cut_data)
            
    return
    