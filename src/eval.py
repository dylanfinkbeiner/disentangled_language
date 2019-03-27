import os
import sys
import subprocess

import torch

from train import predict_relations

from data_utils import conllu_to_sents, sdp_data_loader

CORPORA_DIR = '/corpora/wsj/dependencies'
DATA_DIR = '../data/'
PTB_DEV = os.path.join(CORPORA_DIR, 'treebank.conllu22')
PTB_TEST = os.path.join(CORPORA_DIR, 'treebank.conllu23')
BROWN = ''
GOLD = {
        0 : PTD_DEV,
        1 : PTB_TEST,
        2 : BROWN
        }

def eval(args, parser, data):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mode = args.evalmode

    name = 'DEV' if args.evalmode == 0 else 'TEST'
    #gold = GOLD_DEV if dev else GOLD_TEST
    gold = GOLD[args.evalmode]
    sents_list = conllu_to_sents(gold)

    dataset = data['data_dev'] if dev else data['data_test']
    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    sents_list = conllu_to_sents(gold)
    data_loader = sdp_data_loader(dataset, batch_size=1, shuffle_idx=False)

    predicted = os.path.join(DATA_DIR, name)
    with open(predicted, 'w') as f:
        parser.eval()
        with torch.no_grad():
            for s in sents_list:
                words, pos, sent_len, _, _ = next(data_loader)

                _, S_rel, head_preds = parser(words.to(device), pos.to(device), sent_len)

                rel_preds = predict_relations(S_rel, sent_len)
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
