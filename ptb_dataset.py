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
    '''
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)

    '''

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
        s_split = [line.split('\t') for line in s]
        sents_list[i] = np.array(s_split)[:, mask] #XXX do we really want them to be np arrays at this point?

    conllu_file.close()

    return sents_list


def build_dicts(sents):

    words = []
    pos = []
    rels = []

    for s in sents:
        for line in s:
            words.append(line[0])
            pos.append(line[1])
            rels.append(line[3])

    word_vocab = set(words)
    pos_vocab = set(pos)
    rels_vocab = set(rels)

    word2num = {w:i for (i,w) in enumerate(word_vocab)}
    pos2num = {p:i for (i,p) in enumerate(pos_vocab)}
    rel2num = {r:i for (i,r) in enumerate(rels_vocab)}

    num2word = {i:w for (w,i) in word2num.items()}
    num2pos = {i:p for (p,i) in pos2num.items()}
    num2rel = {i:r for (r,i) in rel2num.items()}


    return word2num, pos2num, rel2num

