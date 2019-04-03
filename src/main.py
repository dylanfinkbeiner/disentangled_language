import sys
import os
import time
import logging
import pickle
#from memory_profiler import profile
from math import ceil
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from args import get_args
from parser import BiaffineParser
from data_utils import build_ptb_dataset
from data_utils import build_brown_dataset
from data_utils import build_dataset_ss

import train
import eval

WEIGHTS_DIR = '../weights'
LOG_DIR = '../log'
DATA_DIR = '../data'
CORPORA_DIR = '/corpora'
DEP_DIR = f'{CORPORA_DIR}/wsj/dependencies'
#BROWN_DIR = f'{CORPORA_DIR}/brown/dependencies'
BROWN_DIR = '../data/brown'
MODEL_NAME = ''
CONLLU_FILES = []
PARANMT_FILE = 'para_100k.txt'
#PARANMT_FILE = 'para_tiny.txt'

PAD_TOKEN = '<pad>' # XXX Weird to have out here
UNK_TOKEN = '<unk>'

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'main.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

if __name__ == '__main__':
    args = get_args()
    init_data = args.initdata
    init_model = args.initmodel
    tmode = args.tmode

    #if not args.semsize + args.semsize == args.hsize:
    #   raise Exception

    d = datetime.datetime.today()
    log.info(f'New session: {d}.\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Filenames
    vocabs_path = os.path.join(DATA_DIR, 'vocabs.pkl')
    data_ptb_path = os.path.join(DATA_DIR, 'data_ptb.pkl')
    data_brown_path = os.path.join(DATA_DIR, 'data_brown.pkl')
    data_ss_path = os.path.join(DATA_DIR, 'data_ss.pkl')

    #init_sdp = not os.path.exists(vocabs_path) or not os.path.exists(data_ptb_path) or not os.path.exists(data_brown_path) or init_data
    init_sdp = False
    #init_ss = (not os.path.exists(data_ss_path) or init_data) and tmode > 0
    init_ss = False
    load_data = not init_sdp and not init_ss

    if init_sdp:
        log.info(f'Initializing SDP data (including vocabs).')
        ptb_conllus = sorted(
                [os.path.join(DEP_DIR, f) for f in os.listdir(DEP_DIR)])
        brown_conllus = [os.path.join(BROWN_DIR, f) for f in os.listdir(BROWN_DIR)]

        data_ptb, x2i, i2x, word_counts = build_ptb_dataset(ptb_conllus)

        data_brown = build_brown_dataset(brown_conllus, x2i=x2i)

        with open(vocabs_path, 'wb') as f:
            pickle.dump((x2i, i2x), f)
        with open(data_ptb_path, 'wb') as f:
            pickle.dump((data_ptb, word_counts), f)
        with open(data_brown_path, 'wb') as f:
            pickle.dump(data_brown, f)

    with open(vocabs_path, 'rb') as f:
        x2i, i2x = pickle.load(f)

    if init_ss:
        log.info(f'Initializing SS data.')
        data_ss = build_dataset_ss(
                os.path.join(DATA_DIR, PARANMT_FILE), x2i)
                #os.path.join(f'{CORPORA_DIR}/paraphrase', PARANMT_FILE), x2i)
        with open(data_ss_path, 'wb') as f:
            pickle.dump(data_ss, f)

    if load_data:
        log.info(f'Loading pickled data.')
        with open(data_ptb_path, 'rb') as f:
            data_ptb, word_counts = pickle.load(f)
        if tmode > 0:
            with open(data_ss_path, 'rb') as f:
                data_ss = pickle.load(f)
        if args.eval:
            with open(data_brown_path, 'rb') as f:
                data_brown = pickle.load(f)

    vocabs = {'x2i': x2i, 'i2x': i2x}

    parser = BiaffineParser(
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            num_relations = len(x2i['rel']),
            hidden_size = args.hsize,
            padding_idx = x2i['word'][PAD_TOKEN],
            unk_idx = x2i['word'][UNK_TOKEN])
    parser.to(device)

    weights_path = os.path.join(WEIGHTS_DIR, args.model)

    if not init_model and os.path.exists(weights_path):
        log.info(f'Loading state dict from: \"{weights_path}\"')
        parser.load_state_dict(torch.load(weights_path))
    else:
        log.info(f'Model will have randomly initialized parameters.')

    if not args.eval:
        data = {'data_ptb' : data_ptb,
                'vocabs' : vocabs,
                'word_counts' : word_counts}
        if tmode > 0:
            data['data_ss'] = data_ss

        train.train(args, parser, data, weights_path=weights_path)

    else:
        data = {'ptb_test': data_ptb['test'],
                'ptb_dev': data_ptb['dev'],
                'brown_cf' : data_brown['cf'],
                'vocabs' : vocabs}

        # Evaluate model
        eval.eval(args, parser, data)

