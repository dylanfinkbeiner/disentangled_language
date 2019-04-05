import os
import sys
import subprocess
import datetime

import torch

from train import predict_relations

from data_utils import conllu_to_sents, sdp_data_loader

WSJ_DIR = '/corpora/wsj/dependencies'
#BROWN_DIR = '/corpora/brown/dependencies'
BROWN_DIR = '../data/brown'
DATA_DIR = '../data/'
PTB_DEV = os.path.join(WSJ_DIR, 'treebank.conllu22')
PTB_TEST = os.path.join(WSJ_DIR, 'treebank.conllu23')
BROWN_CF = os.path.join(BROWN_DIR, 'cf.conllu')
GOLD = {
        0 : PTB_DEV,
        1 : PTB_TEST,
        2 : BROWN_CF
        }

NAMES = {
        0 : 'ptb_dev',
        1 : 'ptb_test',
        2 : 'brown_cf'
        }

def eval(args, parser, data):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mode = args.evalmode

    name = NAMES[mode]
    gold = GOLD[mode]
    sents_list = conllu_to_sents(gold)

    print(f'In eval mode {mode}. Evaluating on {name} dataset.')

    dataset = data[name]
    data_loader = sdp_data_loader(dataset, batch_size=1, shuffle_idx=False, custom_task=False)
    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    predicted = os.path.join(DATA_DIR, name)
    with open(predicted, 'w') as f:
        parser.eval()
        with torch.no_grad():
            for s in sents_list:
                batch = next(data_loader)
                sent_len = batch['sent_lens']

                _, S_rel, head_preds = parser(batch['words'].to(device), batch['pos'].to(device), sent_len)

                rel_preds = predict_relations(S_rel)
                rel_preds = rel_preds.view(-1)
                rel_preds = [i2r[rel] for rel in rel_preds.numpy()]

                s[:,6] = head_preds.view(-1)[1:].cpu().numpy()
                s[:,7] = rel_preds[1:]

                for line in s:
                    f.write('\t'.join(line))
                    f.write('\n')

                f.write('\n')

    # Run official conll17 evaluation script
    print(f'==== RESULTS FOR {name} ====')
    os.system(f'./conll17_ud_eval.py -v {gold} {predicted}')
