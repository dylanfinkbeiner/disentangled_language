import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import subprocess

DATA_PATH = '../data' #TODO

class PTB_Dataset(Dataset):
    def __init__(self, num_sents):
        #Fetch data
        self.data = None

        sent_count = subprocess.check_output(
                f"ls {data_path} | wc -l",
                shell=True)
        sent_count = int(sent_count.strip())

        self.len = None
        self.x_data = None
        self.y_data = None

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]


'''NOTE:
Probably we will call this once for a 
separate conllu file for the train, dev, and test sets
'''
def conllu_to_sents(f: str):

    mask = [1, 4, 6, 7] # [word, pos, head, rel]

    conllu_file = open(f, 'r')
    lines = conllu_file.readlines()

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sents_list = []
    sent_start = 0
        for sent_end in split_points:
            sents_list.append(lines[sent_start: sent_end])
            sent_start = sent_end + 1
    
    for i, s in enumerate(sents_list):
        cleaned = [line.split('\t') for line in s]
        sents_list[i] = np.array(cleaned)[:, mask] #XXX do we really want them to be np arrays at this point?

    conllu_file.close()

    return sents_list
    



