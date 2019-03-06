import os
import sys
import subprocess

import torch

from train import predict_rels

from data_utils import conllu_to_sents


CORPORA_DIR = '/corpora/'
DATA_DIR = '../data/'
GOLD_CONLLU = f'{CORPORA_DIR}/treebank.conllu23'
PREDICTED_CONLLU = '{DATA_DIR}/predicted.conllu'

def eval(args, parser, data):

    sents_list = conllu_to_sents(GOLD_CONLLU)

    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    # Get loader, don't shuffle since we want order to match gold
    data_loader = sdp_data_loader(data['data_test'], batch_size=1, shuffle=False)

    parser.eval()
    with open(PREDICTED_CONLLU, 'w') as f:
        with torch.no_grad():
            for s in sents_list:
                words, pos, sent_len = next(data_loader)

                _, S_rel, head_preds = parser(words, pos, sent_len)

                rel_preds = predict_rels(S_rel, sent_len)
                rel_preds = rel_preds.view(-1)
                rel_preds = [i2r[rel] for rel in rel_preds.numpy()]

                s[:,6] = head_preds.numpy()
                s[:,7] = rel_preds

                for line in s:
                    f.write('\t'.join(line))
                    f.write('\n')

                f.write('\n')
