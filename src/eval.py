import os
import sys
import subprocess
import datetime

import torch

import conll17_ud_eval
from conll17_ud_eval import evaluate, load_conllu

from train import predict_relations

from data_utils import conllu_to_sents, sdp_data_loader, build_sdp_dataset
from utils import predict_relations

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
WSJ_DIR = os.path.join(CORPORA_DIR, 'wsj/dependencies')
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

    exp_path = '_'.join([exp_path_base] + names)
    exp_file = open(exp_path, 'a')
    exp_file.write(f'Training experiment for model : {model_name}')

    vocabs = data['vocabs']
    i2r = vocabs['i2x']['rel']

    names = []
    golds = []
    sents = []

    if not args.ef:
        eval_flags = args.e
        # By default, evaluate on all datasets
        if not eval_flags:
            eval_flags = list(range(len(GOLD)))

        for flag in eval_flags:
            names.append(NAMES[flag])
            gold = GOLD[flag]
            golds.append(gold)
            sents.append(conllu_to_sents(gold))
    else:
        eval_file = args.ef
        name = os.path.splitext(eval_file)[0].split('/')[-1].lower()
        path = os.path.join(DATA_DIR, eval_file)
        names = [name]
        golds = [path]
        sents = [conllu_to_sents(path)]
        data[name] = build_sdp_dataset([path], vocabs['x2i'])[name]

    print(f'Evaluating on datasets: {names}')

    for name, gold, sents_list in zip(names, golds, sents):
        dataset = data[name]
        data_loader = sdp_data_loader(dataset, batch_size=1, shuffle_idx=False)
        predicted = os.path.join(DATA_DIR, name + '_predicted')
        with open(predicted, 'w') as f:
            parser.eval()
            with torch.no_grad():
                for s in sents_list:
                    batch = next(data_loader)
                    sent_len = batch['sent_lens'].to(device)

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

        with open(gold, 'r') as f:
            gold_ud = load_conllu(f)
        with open(predicted, 'r') as f:
            predicted_ud = load_conllu(f)
        evaluation = evaluate(gold_ud, predicted_ud)

        with open(exp_path, 'a') as f:
            info = ("Results for {}:\n LAS : {:10.2f} | UAS : {:10.2f} \n".format(
                name,
                100 * evaluation['UAS'].aligned_accuracy,
                100 * evaluation['LAS'].aligned_accuracy,
            ))
            f.write(info)

        print_results(evaluation, name)

    exp_file.close()


def eval_ss(args):

    for year in years:
        curr_data = data[year]
        predictions = []
        for s1, s2, t in curr_data:
                w1, p1, sl1 = prepare_batch_ss(s1)
                w1, p1, sl1 = prepare_batch_ss(s2)
                h1, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1.to(device))
                h2, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2.to(device))

                predictions.append(predict_sem_sim(h1,h2), args=args)

        corr = sts_score(predictions, targets)

        print(year, corr)



def print_results(evaluation, name):
    print(f"---------------Results for {name} dataset------------------")

    metrics = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "Feats", "AllTags", "Lemmas", "UAS", "LAS"]

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
    print("-----------+-----------+-----------+-----------+-----------")

