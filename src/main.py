import datetime
import logging
from math import ceil
import os
import pickle
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from args import get_args
from data_utils import build_ptb_dataset, build_sdp_dataset, build_ss_dataset
import train
import eval
from parser import BiaffineParser


WEIGHTS_DIR = '../weights'
LOG_DIR = '../log'
DATA_DIR = '../data'
EXPERIMENTS_DIR = '../experiments'

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
#STS_DIR = '/home/AD/dfinkbei/sts'
STS_DIR = f'{DATA_DIR}/sts'
DEP_DIR = f'{CORPORA_DIR}/wsj/dependencies'
#BROWN_DIR = f'{CORPORA_DIR}/brown/dependencies'
BROWN_DIR = '../data/brown'
MODEL_NAME = ''
CONLLU_FILES = []
PARANMT_FILE = 'para_100k.txt'

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
    d = datetime.datetime.now()
    log.info(f'New session: {d}\n')

    args = get_args()
    syn_eval = args.e != None or args.ef != None
    sem_eval = args.es
    evaluating = syn_eval or sem_eval

    # Build experiment file structure
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.model)
    if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
            os.mkdir(os.path.join(exp_dir, 'training'))
            os.mkdir(os.path.join(exp_dir, 'evaluation'))

    exp_type = 'evaluation' if evaluating else 'training'

    day = f'{d:%m_%d_%Y}'
    day_dir = os.path.join(exp_dir, exp_type, day)
    if not os.path.isdir(day_dir):
        os.mkdir(day_dir)

    exp_path_base = os.path.join(day_dir, f'{d:%H%M}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Filenames
    vocabs_path = os.path.join(DATA_DIR, 'vocabs.pkl')
    data_ptb_path = os.path.join(DATA_DIR, 'data_ptb.pkl')
    data_brown_path = os.path.join(DATA_DIR, 'data_brown.pkl')
    #data_ss_path = os.path.join(DATA_DIR, 'data_ss.pkl')
    data_ss_path = os.path.join(DATA_DIR, 'data_ss_local.pkl')

    #init_sdp = (not os.path.exists(vocabs_path)
    #        or not os.path.exists(data_ptb_path) 
    #        or not os.path.exists(data_brown_path) or args.init_sdp)
    init_sdp = False
    #init_ss = (not os.path.exists(data_ss_path) or args.initdata) and args.train_mode > 0
    init_ss = False # NOTE must stay this way until we get CoreNLP working on pitts

    if init_sdp:
        log.info(f'Initializing syntactic dependency parsing data (including vocabs).')
        ptb_conllus = sorted(
                [os.path.join(DEP_DIR, f) for f in os.listdir(DEP_DIR)])
        brown_conllus = [os.path.join(BROWN_DIR, f) for f in os.listdir(BROWN_DIR)]

        data_ptb, x2i, i2x, word_counts = build_ptb_dataset(ptb_conllus)

        data_brown = build_sdp_dataset(brown_conllus, x2i=x2i)

        with open(vocabs_path, 'wb') as f:
            pickle.dump((x2i, i2x), f)
        with open(data_ptb_path, 'wb') as f:
            pickle.dump((data_ptb, word_counts), f)
        with open(data_brown_path, 'wb') as f:
            pickle.dump(data_brown, f)
    else: 
        log.info(f'Loading pickled syntactic dependency parsing data.')
        with open(data_ptb_path, 'rb') as f:
            data_ptb, word_counts = pickle.load(f)
        with open(vocabs_path, 'rb') as f:
            x2i, i2x = pickle.load(f)
        if evaluating:
            with open(data_brown_path, 'rb') as f:
                data_brown = pickle.load(f)

    if init_ss:
        log.info(f'Initializing semantic similarity data.')
        data_ss = {}
        STS_INPUT = os.path.join(STS_DIR, 'input')
        STS_GS = os.path.join(STS_DIR, 'gs')

        #train_ss = build_ss_dataset(os.path.join(DATA_DIR, PARANMT_FILE), gs='', x2i=x2i)
        dev_ss = build_ss_dataset(
                os.path.join(STS_INPUT, '2017'),
                gs=os.path.join(STS_GS, '2017'),
                x2i=x2i)
        test_ss = {}
        years = os.listdir(STS_INPUT)
        years.remove('2017')
        for year in years:
            test_ss[year] = build_ss_dataset(
                os.path.join(STS_INPUT, year),
                gs=os.path.join(STS_GS, year),
                x2i=x2i)

        #data_ss['train'] = train_ss
        data_ss['dev'] = dev_ss
        data_ss['test'] = test_ss
        with open(data_ss_path, 'wb') as f:
            pickle.dump(data_ss, f)
    elif args.train_mode > 0 or args.es:
        log.info(f'Loading pickled semantic similarity data.')
        with open(data_ss_path, 'rb') as f:
            data_ss = pickle.load(f)


    vocabs = {'x2i': x2i, 'i2x': i2x}

    parser = BiaffineParser(
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            num_relations = len(x2i['rel']),
            hidden_size = args.h_size,
            padding_idx = x2i['word'][PAD_TOKEN],
            unk_idx = x2i['word'][UNK_TOKEN])
    parser.to(device)

    weights_path = os.path.join(WEIGHTS_DIR, args.model)

    if (not args.initmodel) and os.path.exists(weights_path):
        log.info(f'Loading state dict from: \"{weights_path}\"')
        parser.load_state_dict(torch.load(weights_path))
    else:
        log.info(f'Model will have randomly initialized parameters.')
        args.initmodel = True

    if not evaluating:
        data = {'data_ptb' : data_ptb,
                'vocabs' : vocabs,
                'word_counts' : word_counts}
        if args.train_mode > 0:
            data['data_ss'] = data_ss

        train.train(args, parser, data, weights_path=weights_path, exp_path_base=exp_path_base)

    else:
        if syn_eval:
            data = {'ptb_test': data_ptb['test'],
                    'ptb_dev': data_ptb['dev'],
                    'brown_cf': data_brown['cf'],
                    'vocabs': vocabs}
            eval.eval_sdp(args, parser, data, exp_path_base=exp_path_base)

        if sem_eval:
            data = {'semeval': data_ss['test'],
                    'vocabs': vocabs}
            eval.eval_sts(args, parser, data, exp_path_base=exp_path_base)

