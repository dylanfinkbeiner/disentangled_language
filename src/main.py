import sys
import os
import time
import logging
import pickle
from memory_profiler import profile
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from parser import BiaffineParser
from data_utils import get_dataset_sdp, sdp_data_loader, word_dropout
from data_utils import get_dataset_ss, ss_data_loader, prepare_batch_ss
from args import get_args

import train
import eval

WEIGHTS_DIR = '../weights'
LOG_DIR = '../log'
DATA_DIR = '../data'
CORPORA_DIR = '/corpora'
MODEL_NAME = ''
CONLLU_FILES = []
#CONLLU_FILE = 'tenpercentsample.conllu'
PARANMT_FILE = 'para_tiny.txt'

PAD_TOKEN = '<pad>' # XXX Weird to have out here

if __name__ == '__main__':
    args = get_args()

    init_data = args.initdata
    init_model = args.initmodel

    # Filenames
    vocabs_path = os.path.join(DATA_DIR, 'vocabs.pkl')
    data_sdp_path = os.path.join(DATA_DIR, 'data_sdp.pkl')
    data_ss_path = os.path.join(DATA_DIR, 'data_ss.pkl')

    if not os.path.exists(vocabs_path) \
            or not os.path.exists(data_sdp_path) \
            or not os.path.exists(data_ss_path):
        init_data = True

    if init_data:
        conllu_files = [f for f in os.listdir(f'{CORPORA_DIR}/wsj')]

        data_sdp, x2i, i2x, word_counts = get_dataset_sdp(conllu_files, training=True)
        data_ss = get_dataset_ss(os.path.join(f'{CORPORA_DIR}/paraphrase', PARANMT_FILE), x2i)

        with open(vocabs_path, 'wb') as f:
            pickle.dump((x2i, i2x), f)
        with open(data_sdp_path, 'wb') as f:
            pickle.dump((data_sdp, word_counts), f)
        with open(data_ss_path, 'wb') as f:
            pickle.dump((data_ss), f)
    else:
        with open(vocabs_path, 'rb') as f:
            x2i, i2x = pickle.load(f)
        with open(data_sdp_path, 'rb') as f:
            data_sdp, word_counts = pickle.load(f)
        with open(data_ss_path, 'rb') as f:
            data_ss = pickle.load(f)

    vocabs = {'x2i': x2i, 'i2x': i2x}

    parser = BiaffineParser(
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            num_relations = len(x2i['rel']),
            hidden_size = args.hidden_size,
            padding_idx = x2i['word'][PAD_TOKEN])
    parser.to(device)

    if not os.path.isdir(WEIGHTS_DIR)
        os.makedirs(WEIGHTS_DIR)

    weights_path = os.path.join(WEIGHTS_DIR, args.model)

    if not init_model and os.path.exists(weights_path):
        parser.load_state_dict(torch.load(weights_path))

    if not args.eval:
        data = {'data_sdp' : data_sdp,
                'data_ss' : data_ss,
                'vocabs' : vocabs,
                'word_counts' : word_counts}

        train(args, parser, data, weights_path=weights_path)

    else:
        data = {'data_test', data_sdp['test'],
                'vocabs' : vocabs}

        # Evaluate model
        eval(args, parser, data)
