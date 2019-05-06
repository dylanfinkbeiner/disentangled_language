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
import data_utils
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
#PARANMT_FILE = 'para_100k.txt'
PARANMT_FILE = 'para-nmt-5m-processed.txt'
CHUNKS_DIR = os.path.join(DATA_DIR, '5m')
POS_ONLY = False
POS = False

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
    if args.syn_size > args.h_size:
        print('Syn/hidden mismatch')
        exit()

    syn_eval = args.e != None or args.ef != None
    evaluating = syn_eval or args.evaluate_semantic

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

    # possible?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Filenames
    vocabs_path = os.path.join(DATA_DIR, 'vocabs.pkl')
    data_ptb_path = os.path.join(DATA_DIR, 'data_ptb.pkl')
    data_brown_path = os.path.join(DATA_DIR, 'data_brown.pkl')
    #data_ss_path = os.path.join(DATA_DIR, 'data_ss.pkl')
    data_ss_dir = os.path.join(DATA_DIR, 'data_ss_local')
    if not os.path.isdir(data_ss_dir):
        os.mkdir(data_ss_dir)

    #init_sdp = (not os.path.exists(vocabs_path)
    #        or not os.path.exists(data_ptb_path) 
    #        or not os.path.exists(data_brown_path) or args.init_sdp)
    init_sdp = False
    #init_ss = (not os.path.exists(data_ss_path) or args.initdata) and args.train_mode > 0
    #init_ss = False # NOTE must stay this way until we get CoreNLP working on pitts
    init_ss = args.init_ss

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

    data_ss = {}
    STS_INPUT = os.path.join(STS_DIR, 'input')
    STS_GS = os.path.join(STS_DIR, 'gs')

    if not os.path.isdir(os.path.join(CHUNKS_DIR, 'pkl')):
        os.mkdir(os.path.join(CHUNKS_DIR, 'pkl'))
    if not os.path.isdir(os.path.join(CHUNKS_DIR, 'tagged')):
        os.mkdir(os.path.join(CHUNKS_DIR, 'tagged'))
    if not os.path.isdir(os.path.join(STS_DIR, 'tagged')):
        os.mkdir(os.path.join(STS_DIR, 'tagged'))

    train_ss = {'sent_pairs': [], 'targets': []}
    if 'train' in init_ss:
        log.info(f'Initializing SS train data.')
        chunks = sorted(list(os.listdir(CHUNKS_DIR)))
        chunks_clean = []
        for chunk in chunks:
            if os.path.splitext(chunk)[-1] == '.txt':
                chunks_clean.append(chunk)

        for chunk in chunks_clean:
            print(f'Processing chunk {chunk}')
            if POS:
                raw_sents_path = os.path.join(CHUNKS_DIR, 'tagged', f'{os.path.splitext(chunk)[0]}-tagged.pkl')
    
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(CHUNKS_DIR, chunk))
    
                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
    
            if not POS_ONLY:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)

                x2i, i2x = build_dicts(raw_se

                train_path = os.path.join(CHUNKS_DIR, 'pkl', f'{os.path.splitext(chunk)[0]}.pkl')
                if os.path.exists(train_path):
                    if input('Path to data for chunk {chunk} exists. Overwrite? [y/n] ').lower() != 'y': 
                        continue
                train_ss = build_ss_dataset(raw_sent_pairs, gs='', x2i=x2i)
                with open(train_path, 'wb') as pkl:
                    pickle.dump(train_ss, pkl)


    elif args.train_mode > 0:
        log.info(f'Loading pickled SS train data.')
        chunks_pkl = sorted(list(os.listdir(os.path.join(CHUNKS_DIR, 'pkl'))))

        for chunk_pkl in chunks_pkl[:args.n_chunks]:
            train_path = os.path.join(CHUNKS_DIR, 'pkl', chunk_pkl)
            with open(train_path, 'rb') as pkl:
                curr = pickle.load(pkl)
                train_ss['sent_pairs'].extend(curr['sent_pairs'])
                train_ss['targets'].extend(curr['targets'])

    dev_ss = {}
    dev_path = os.path.join(data_ss_dir, 'ss_dev.pkl')
    if 'dev' in init_ss:
        log.info(f'Initializing SS dev data.')
        
        raw_sents_path = os.path.join(STS_DIR, 'tagged', '2017-tagged.pkl')
        raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(STS_INPUT, '2017'))

        with open(raw_sents_path, 'wb') as pkl:
            pickle.dump(raw_sent_pairs, pkl)

        if not POS_ONLY:
            with open(raw_sents_path, 'rb') as pkl:
                raw_sent_pairs = pickle.load(pkl)
            dev_ss = build_ss_dataset(
                raw_sent_pairs,
                gs=os.path.join(STS_GS, '2017'),
                x2i=x2i)
            with open(dev_path, 'wb') as pkl:
                pickle.dump(dev_ss, pkl)
    elif args.train_mode > 0:
        log.info(f'Loading pickled SS dev data.')
        with open(dev_path, 'rb') as pkl:
            dev_ss = pickle.load(pkl)

    test_ss = {}
    test_path = os.path.join(data_ss_dir, 'ss_test.pkl')
    if 'test' in init_ss:
        log.info(f'Initializing SS test data.')

        years = os.listdir(STS_INPUT)
        for year in years:
            raw_sents_path = os.path.join(STS_DIR, 'tagged', f'{year}-tagged.pkl')
            if POS:
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(STS_INPUT, year))
                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
            if not POS_ONLY:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)
                test_ss[year] = build_ss_dataset(
                    raw_sent_pairs,
                    gs=os.path.join(STS_GS, year),
                    x2i=x2i)
                with open(test_path, 'wb') as pkl:
                    pickle.dump(test_ss, pkl)
    elif args.evaluate_semantic:
       log.info(f'Loading pickled SS test data.')
       with open(test_path, 'rb') as pkl:
           test_ss = pickle.load(pkl)


    data_ss['train'] = train_ss
    data_ss['dev'] = dev_ss
    data_ss['test'] = test_ss

    vocabs = {'x2i': x2i, 'i2x': i2x}

    parser = BiaffineParser(
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            num_relations = len(x2i['rel']),
            hidden_size = args.h_size,
            padding_idx = x2i['word'][PAD_TOKEN],
            unk_idx = x2i['word'][UNK_TOKEN],
            device=device)
    print(parser.BiLSTM.init_we.device)
    print(parser.BiLSTM.word_emb.weight.device)
#    breakpoint()

    weights_path = os.path.join(WEIGHTS_DIR, args.model)

    if (not args.init_model) and os.path.exists(weights_path):
        log.info(f'Loading state dict from: \"{weights_path}\"')
        parser.load_state_dict(torch.load(weights_path))
    else:
        log.info(f'Model will have randomly initialized parameters.')
        args.init_model = True

    if not evaluating:
        args.epochs = args.epochs if args.train_mode != -1 else 1
        data = {'data_ptb' : data_ptb,
                'vocabs' : vocabs,
                'word_counts' : word_counts,
                'device': device}
        if args.train_mode > 0:
            data['data_ss'] = data_ss

        train.train(args, parser, data, weights_path=weights_path, exp_path_base=exp_path_base)

    else:
        if syn_eval:
            data = {'ptb_test': data_ptb['test'],
                    'ptb_dev': data_ptb['dev'],
                    'brown_cf': data_brown['cf'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sdp(args, parser, data, exp_path_base=exp_path_base)

        if args.evaluate_semantic:
            data = {'semeval': data_ss['test'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sts(args, parser, data, exp_path_base=exp_path_base)

