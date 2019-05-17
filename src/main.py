import datetime
import pytz
import logging
import os
import pickle
import time

import torch

from args import get_args
#from data_utils import build_ptb_dataset, build_sdp_dataset, build_ss_dataset
import data_utils
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
POS_ONLY = False

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
    d = datetime.datetime.utcnow()
    d = d.astimezone(pytz.timezone("America/Los_Angeles"))
    log.info(f'New session: {d}\n')

    args = get_args()
    if args.syn_size > args.h_size:
        print('Syn/hidden mismatch')
        exit()

    syn_eval = args.e != None or args.ef != None
    evaluating = syn_eval or args.evaluate_semantic

    # Experiment logistics
    experiment = {}
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.model)
    if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
    exp_type = 'evaluation' if evaluating else 'training'
    day = f'{d:%m_%d}'
    exp_path = os.path.join(exp_dir, '_'.join([exp_type, day]) + '.txt')
    experiment['dir'] = exp_dir
    experiment['path'] = exp_path
    experiment['date'] = d

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    paths = DataPaths(filtered=args.filter,
            use_paragram=args.sl999, 
            glove_d=args.glove_d)

    # Populate syntactic dependency parsing data
    log.info(f'Loading pickled syntactic dependency parsing data.')
    with open(paths.data_ptb, 'rb') as pkl:
        data_ptb, word_counts = pickle.load(pkl)
    with open(paths.ptb_vocabs, 'rb') as pkl:
        x2i, i2x = pickle.load(pkl)
    if evaluating:
        with open(paths.data_brown, 'rb') as pkl:
            data_brown = pickle.load(pkl)

    args.using_pretrained = False
    if args.sl999 or args.glove_d:
        args.using_pretrained = True
        path = paths.sl999_data if args.sl999 else paths.glove_data
        log.info(f'Loading pretrained embedding data from {path}')
        with open(path, 'rb') as pkl:
            embedding_data = pickle.load(pkl)
        x2i['word'] = embedding_data['w2i']
        i2x['word'] = embedding_data['i2w']
        pretrained_e = embedding_data['word_e']

    # Populate semantic similarity data
    data_ss = {}

    ss_train = {'sent_pairs': [], 'targets': []}
    if args.train_mode > 0:
        log.info(f'Loading pickled SS train data.')

        chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
        for chunk in chunks_txt[3:args.n_chunks]:
            train_path_chunk = paths.ss_train_base + f'{os.path.splitext(chunk)[0]}.pkl'
            with open(train_path_chunk, 'rb') as pkl:
                curr = pickle.load(pkl)
                ss_train['sent_pairs'].extend(curr['sent_pairs'])
                ss_train['targets'].extend(curr['targets'])
        data_ss['train'] = ss_train

    ss_test = {}
    if args.evaluate_semantic or args.train_mode > 0:
       log.info(f'Loading pickled SS test data.')
       with open(paths.ss_test, 'rb') as pkl:
           ss_test = pickle.load(pkl)
       data_ss['dev'] = ss_test['2017']
       data_ss['test'] = ss_test

    # Prepare parser
    parser = BiaffineParser(
            word_e_size = pretrained_e.shape[-1] if args.using_pretrained else args.we,
            pos_e_size = args.pe,
            pretrained_e = pretrained_e if args.using_pretrained else None,
            word_vocab_size = len(x2i['word']),
            pos_vocab_size = len(x2i['pos']),
            num_relations = len(x2i['rel']),
            hidden_size = args.h_size,
            lstm_layers = args.lstm_layers,
            padding_idx = x2i['word'][PAD_TOKEN],
            unk_idx = x2i['word'][UNK_TOKEN],
            device = device)

    weights_path = os.path.join(WEIGHTS_DIR, args.model)

    if os.path.exists(weights_path):
        log.info(f'Loading state dict from: \"{weights_path}\"')
        parser.load_state_dict(torch.load(weights_path))
        args.init_model = False
    else:
        log.info(f'Model will have randomly initialized parameters.')
        args.init_model = True

    vocabs = {'x2i': x2i, 'i2x': i2x}

    if not evaluating:
        args.epochs = args.epochs if args.train_mode != -1 else 1
        data = {'data_ptb' : data_ptb,
                'vocabs' : vocabs,
                'word_counts' : word_counts,
                'device': device}
        if args.train_mode > 0:
            data['data_ss'] = data_ss

        train.train(args, parser, data, weights_path=weights_path, experiment=experiment)

    else:
        if syn_eval:
            data = {'ptb_test': data_ptb['test'],
                    'ptb_dev': data_ptb['dev'],
                    'brown_cf': data_brown['cf'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sdp(args, parser, data, experiment=experiment)

        if args.evaluate_semantic:
            data = {'semeval': data_ss['test'],
                    'device': device,
                    'vocabs': vocabs}
            eval.eval_sts(args, parser, data, experiment=experiment)

