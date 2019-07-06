import datetime as dt
import pytz
import logging
import os
import pickle
import time
from collections import namedtuple
import random
random.seed(1)

import torch
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(1)

from args import get_args
from parser import BiaffineParser
import train
import eval
from preprocess import DataPaths


WEIGHTS_DIR = '../weights'
LOG_DIR = '../log'
DATA_DIR = '../data'
EXPERIMENTS_DIR = '../experiments'

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
STS_DIR = f'{DATA_DIR}/sts'
DEP_DIR = f'{CORPORA_DIR}/wsj/dependencies'
#BROWN_DIR = f'{CORPORA_DIR}/brown/dependencies'
BROWN_DIR = '../data/brown'
PARANMT_DIR = os.path.join(DATA_DIR, 'paranmt_5m')

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


def save_permanent_params(exp_dir, args):
    txt_path = os.path.join(exp_dir, 'parameters.txt')
    args_dict = dict(vars(args))
    with open(txt_path, 'w') as f:
        for k, v in args_dict.items():
            f.write(f'{k} : {v} \n')

    pkl_path = os.path.join(exp_dir, 'parameters.pkl')
    with open(pkl_path, 'wb') as pkl:
        pickle.dump(args_dict, pkl)


def main():
    d = dt.datetime.now().astimezone(pytz.timezone("America/Los_Angeles"))
    log.info(f'New session: {d}\n')

    args = get_args()

    syn_eval = args.e != None or args.ef != None
    evaluating = syn_eval or args.evaluate_semantic or args.evaluate_stag

    # Experiment logistics
    experiment = {}
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.model)
    if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
            save_permanent_params(exp_dir, args)
    exp_type = 'evaluation' if evaluating else 'training'
    day = f'{d:%m_%d}'
    exp_path = os.path.join(exp_dir, '_'.join([exp_type, day]) + '.txt')
    experiment['dir'] = exp_dir
    experiment['path'] = exp_path
    experiment['date'] = d

    if evaluating:
        pkl_path = os.path.join(exp_dir, 'parameters.pkl')
        if os.path.exists(pkl_path):
            with open(os.path.join(exp_dir, 'parameters.pkl'), 'rb') as pkl:
                args_dict = pickle.load(pkl)
            args_dict['e'] = args.e
            args_dict['ef'] = args.ef
            args_dict['evaluate_semantic'] = args.evaluate_semantic
            args_dict['evaluate_stag'] = args.evaluate_stag
            args = namedtuple('args', args_dict.keys())(*args_dict.values())



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    paths = DataPaths(filtered=args.filter, glove_d=args.glove_d)

    # Populate syntactic dependency parsing data
    log.info(f'Loading pickled syntactic dependency parsing data.')
    with open(paths.data_ptb, 'rb') as pkl:
        data_ptb, word_counts = pickle.load(pkl)
    with open(paths.ptb_vocabs, 'rb') as pkl:
        x2i, i2x = pickle.load(pkl)
    if evaluating:
        with open(paths.data_brown, 'rb') as pkl:
            data_brown = pickle.load(pkl)

    if args.glove_d:
        path = paths.glove_data
        log.info(f'Loading pretrained embedding data from {path}')
        with open(path, 'rb') as pkl:
            embedding_data = pickle.load(pkl)
        x2i['word'] = embedding_data['w2i']
        print(len(x2i['word']))
        i2x['word'] = embedding_data['i2w']
        pretrained_e = embedding_data['word_e']

    # Populate semantic similarity data
    data_ss = {}

    ss_train = {'sent_pairs': [], 'targets': []}
    #if args.evaluate_semantic or args.train_mode > 0:
    if args.evaluate_semantic or 1 in args.train_mode:
        log.info(f'Loading pickled semantic similarity data.')

        chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
        for chunk in chunks_txt[:args.n_chunks]:
            train_path_chunk = paths.ss_train_base + f'{os.path.splitext(chunk)[0]}.pkl'
            with open(train_path_chunk, 'rb') as pkl:
                curr = pickle.load(pkl)
                ss_train['sent_pairs'].extend(curr['sent_pairs'])
                ss_train['targets'].extend(curr['targets'])
        data_ss['train'] = ss_train

        ss_test = {}
        log.info(f'Loading pickled SS test data.')
        with open(paths.ss_test, 'rb') as pkl:
            ss_test = pickle.load(pkl)
            data_ss['dev'] = ss_test['2017']
            data_ss['test'] = ss_test
    

    #stag_flag = args.evaluate_stag or (args.train_mode > 3)
    stag_flag = args.evaluate_stag or 2 in args.train_mode
    if stag_flag:
        data_stag = {}
        log.info('Loading pickled supertagging training data.')
        with open(paths.data_stag, 'rb') as pkl:
            data_stag = pickle.load(pkl)
        with open(paths.stag_vocabs, 'rb') as pkl:
            s2i, i2s = pickle.load(pkl)
        x2i['stag'] = s2i
        i2x['stag'] = i2s


    # Prepare parser
    parser = BiaffineParser(
            word_e_size = pretrained_e.shape[-1] if args.glove_d else args.we,
            pos_e_size = args.pe if args.pe > 0 else None,
            pretrained_e = pretrained_e if args.glove_d else None,
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            stag_vocab_size = len(x2i['stag']) if stag_flag else None,
            syn_h = args.syn_h,
            sem_h = args.sem_h,
            final_h = args.final_h,
            syn_nlayers = args.syn_nlayers,
            sem_nlayers = args.sem_nlayers,
            final_nlayers = args.final_nlayers,
            embedding_dropout=args.embedding_dropout,
            lstm_dropout=args.lstm_dropout,
            d_arc=500,
            d_rel=100,
            num_relations = len(x2i['rel']),
            arc_dropout=0.33,
            rel_dropout=0.33,
            padding_idx = x2i['word'][PAD_TOKEN],
            unk_idx = x2i['word'][UNK_TOKEN],
            #word_avg=args.word_avg,
            #vanilla=(args.train_mode == 0),
            vanilla=(0 in args.train_mode and len(args.train_mode) == 1),
            device = device)


    weights_path = os.path.join(WEIGHTS_DIR, args.model)
    if os.path.exists(weights_path):
        log.info(f'Loading state dict from: \"{weights_path}\"')
        parser.load_state_dict(torch.load(weights_path))
    else:
        log.info(f'Model will have randomly initialized parameters.')

    if args.w:
        ih = parser.FinalRNN.lstm.weight_ih_l0.data
        syn_weights = ih[:, :2*args.syn_h]
        sem_weights = ih[:, 2*args.syn_h:]
        print(f'Syn: {syn_weights.norm(2)}')
        print(f'Sem: {sem_weights.norm(2)}')
        breakpoint()

    vocabs = {'x2i': x2i, 'i2x': i2x}

    if not evaluating:
        #args.epochs = args.epochs if args.train_mode != -1 else 1
        args.epochs = args.epochs 
        data = {'data_ptb' : data_ptb,
                'vocabs' : vocabs,
                'word_counts' : word_counts,
                'device': device}
        #if args.train_mode > 0:
        if 1 in args.train_mode:
            data['data_ss'] = data_ss
        #if args.train_mode > 3:
        if 2 in args.train_mode:
            data['data_stag'] = data_stag

        train.train(args, parser, data, weights_path=weights_path, experiment=experiment)

    else:
        if syn_eval:
            data = {'ptb_dev': data_ptb['dev'],
                    'ptb_test': data_ptb['test'],
                    'brown_cf': data_brown['cf'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sdp(args, parser, data, experiment=experiment)

        if args.evaluate_semantic:
            data = {'semeval': data_ss['test'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sts(args, parser, data, experiment=experiment)

        if args.evaluate_stag:
            data = {'dev': data_stag['dev'],
                    'test': data_stag['test'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_stag(args, parser, data, experiment=experiment)

if __name__ == '__main__':
    main()
