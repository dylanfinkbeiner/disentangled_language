import os
import sys
import subprocess
import datetime

import torch

import conll17_ud_eval
from conll17_ud_eval import evaluate, load_conllu

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

def eval(args, parser, data, exp_path_base=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not exp_path_base:
        print('Base of path to experiment documentation file missing.')
        raise Exception

    eval_flags = args.evalflags

    names = []
    golds = []
    sents = []

    # Evaluate on all datasets
    if not eval_flags:
        names = list(NAMES.values())
        gold = list(GOLD.values())

    else:
        for flag in eval_flags:
            names.append(NAMES[flag])
            golds.append(GOLD[flag])
            sents.append(conllu_to_sents(gold))

    print(f'Evaluating on datasets: {names}')

    exp_path = '_'.join([exp_path_base] + names)
    
    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    for name, gold, sents_list in zip(names, golds, sents):
        dataset = data[name]
        data_loader = sdp_data_loader(dataset, batch_size=1, shuffle_idx=False)

        predicted = os.path.join(DATA_DIR, name)
        with open(predicted, 'w') as f:
            parser.eval()
            with torch.no_grad():
                for s in sents_list:
                    batch = next(data_loader)
                    sent_len = batch['sent_lens']

                    _, S_rel, head_preds = parser(batch['words'].to(device), batch['pos'].to(device), sent_len)
                    rel_preds = predict_relations(S_rel, sent_len)
                    rel_preds = rel_preds.view(-1)
                    rel_preds = [i2r[rel] for rel in rel_preds.numpy()]

                    s[:,6] = head_preds.view(-1)[1:].cpu().numpy()
                    s[:,7] = rel_preds[1:]

                    for line in s:
                        f.write('\t'.join(line))
                        f.write('\n')

                    f.write('\n')

        gold_ud = load_conllu(gold)
        predicted_ud = load_conllu(predicted)
        evaluation = evaluate(gold_ud, predicted_ud)

        with open(exp_path, 'a') as f:
            info = ("Results for {name}:\n LAS : {:10.2f} | UAS : {:10.2f} \n".format(
                100 * evaluation['UAS'].accuracy,
                100 * evaluation['LAS'].accuracy,
            ))
            f.write(info)

        print_results(evaluation)


def print_results(evaluation):
    metrics = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "Feats", "AllTags", "Lemmas", "UAS", "LAS"]
    if args.weights is not None:
        metrics.append("WeightedLAS")

    print("Metrics    | Precision |    Recall |  F1 Score | AligndAcc")
    print("-----------+-----------+-----------+-----------+-----------")
    for metric in metrics:
        print("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
            metric,
            100 * evaluation[metric].precision,
            100 * evaluation[metric].recall,
            100 * evaluation[metric].f1,
            "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""
        ))

if __name__ == '__main__':
    eval(None, None, None, exp_path_base='')
