import os
import sys
import subprocess

import torch

from train import predict_relations

from data_utils import conllu_to_sents, sdp_data_loader

CORPORA_DIR = '/corpora/wsj/dependencies'
DATA_DIR = '../data/'
GOLD_CONLLU = os.path.join(CORPORA_DIR, 'treebank.conllu23')
PREDICTED_CONLLU = os.path.join(DATA_DIR, 'predicted.conllu')

def eval(args, parser, data):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sents_list = conllu_to_sents(GOLD_CONLLU)

    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    # Get loader, don't shuffle since we want order to match gold
    data_loader = sdp_data_loader(data['data_test'], batch_size=1, shuffle_idx=False)

    parser.eval()
    with open(PREDICTED_CONLLU, 'w') as f:
        with torch.no_grad():
            for s in sents_list:
                words, pos, sent_len, _, _ = next(data_loader)

                _, S_rel, head_preds = parser(words.to(device), pos.to(device), sent_len)

                rel_preds = predict_relations(S_rel, sent_len)
                rel_preds = rel_preds.view(-1)
                rel_preds = [i2r[rel] for rel in rel_preds.numpy()]

                s[:,6] = head_preds.cpu().numpy()
                s[:,7] = rel_preds

                for line in s:
                    f.write('\t'.join(line))
                    f.write('\n')

                f.write('\n')
