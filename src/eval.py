from math import ceil
import os
import sys
import subprocess
import datetime
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
import torch
import numpy as np

import conll17_ud_eval
from conll17_ud_eval import evaluate, load_conllu

import data_utils
import utils
from parser import unsort

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
WSJ_DIR = os.path.join(CORPORA_DIR, 'wsj')
#BROWN_DIR = '/corpora/brown/dependencies'
BROWN_DIR = '../data/brown'
DATA_DIR = '../data/'
PREDICTED_DIR = os.path.join(DATA_DIR, 'predicted_conllus')

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

YEARS = ['2012', '2013', '2014', '2015', '2016', '2017']


def eval_sdp(args, parser, data, experiment=None):
    pos_tag_eval = False
    device = data['device']

    sdp_eval_data = {}

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
            sents.append(data_utils.conllu_to_sents(gold))
    else:
        eval_file = args.ef
        name = os.path.splitext(eval_file)[0].split('/')[-1].lower()
        path = os.path.join(DATA_DIR, eval_file)
        names = [name]
        golds = [path]
        sents = [conllu_to_sents(path)]
        data[name] = data_utils.build_sdp_dataset([path], vocabs['x2i'])[name]


    with open(experiment['path'], 'a') as exp_file:
        d = experiment['date']
        prelude = '\n' * 3 + f'New syntactic evaluation experiment for model : {args.model}'
        prelude += f'\nStarting date/time: {d:%m %d} at {d:%H:%M:%S}\n'
        print(f'\nEvaluating on datasets: {names}\n')
        exp_file.write(prelude)
        print(prelude)

        total_predictions = 0
        total_correct = 0
        for name, gold, sents_list in zip(names, golds, sents):
            dataset = data[name]
            data_loader = data_utils.sdp_data_loader(dataset, batch_size=1, shuffle_idx=False)
            predicted = os.path.join(PREDICTED_DIR, name + '_predicted.conllu')
            with open(predicted, 'w') as f:
                parser.eval()
                with torch.no_grad():
                    for s in tqdm(sents_list, ascii=True, desc=f'Writing predicted conllu for {name} dataset', ncols=80):
                        batch = next(data_loader)
                        sent_len = batch['sent_lens'].to(device)

                        _, S_rel, head_preds = parser(
                                batch['words'].to(device), 
                                sent_len, 
                                pos=batch['pos'].to(device) if parser.pos_in else None)
                        rel_preds = utils.predict_relations(S_rel)
                        rel_preds = rel_preds.view(-1)
                        rel_preds = [i2r[rel] for rel in rel_preds.numpy()]

                        s[:,6] = head_preds.view(-1)[1:].cpu().numpy()
                        s[:,7] = rel_preds[1:]

                        for line in s:
                            f.write('\t'.join(line))
                            f.write('\n')

                        f.write('\n')

                        if pos_tag_eval:
                            #NOTE still haven't figured out the head thing
                            lstm_input, indices, lens_sorted, _ = parser.Embeddings(batch['words'].to(device), sent_len)
                            #lstm_input, indices, lens_sorted = parser.Embeddings(batch['words'].to(device), sent_len, pos=batch['pos'].to(device))
                            outputs = parser.SyntacticRNN(lstm_input)
                            logits = parser.POSMLP(outputs)
                            predictions = torch.argmax(logits, -1)
                            #breakpoint() # Check that predictions and pos have same shape
                            n_correct = torch.eq(predictions, batch['pos'].to(device)).sum().item()
                            #breakpoint()
                            total_correct += n_correct
                            total_predictions += sent_len.item()

            if pos_tag_eval:
                pos_accuracy = total_correct / total_predictions
                pos_res = f'\nPOS tagging accuracy on {name} is {pos_accuracy * 100}\n'
                exp_file.write(pos_res)
                print(pos_res)

            with open(gold, 'r') as f:
                gold_ud = load_conllu(f)
            with open(predicted, 'r') as f:
                predicted_ud = load_conllu(f)
            evaluation = evaluate(gold_ud, predicted_ud)
            UAS = evaluation['UAS'].aligned_accuracy
            LAS = evaluation['LAS'].aligned_accuracy

            info = '\nResults for {}:\n UAS : {:10.2f} | LAS : {:10.2f} \n'.format(
                name,
                100 * UAS,
                100 * LAS
            )
            exp_file.write(info)
            sdp_eval_data[name] = (UAS, LAS)

            print_results(evaluation, name)

    #weights = {}
    #syn = {}
    #sem = {}
    #ih = parser.FinalRNN.lstm.weight_ih_l0.data
    #syn_weights = ih[:, :2*args.syn_h]
    #sem_weights = ih[:, 2*args.syn_h:]
    #syn['final'] = syn_weights.norm(2).item()
    #sem['final'] = sem_weights.norm(2).item()
    #if parser.SemanticRNN is None:
    #    pass
    #else:
    #    ih = parser.SemanticRNN.lstm.weight_ih_l0.data
    #    pos = ih[:, args.we:] 
    #    word = ih[:, :args.we]
    #    sem['pos'] = pos.norm(2).item() if parser.pos_in else None
    #    sem['word'] = word.norm(2).item()
    #ih = parser.SyntacticRNN.lstm.weight_ih_l0.data
    #pos = ih[:, args.we:]
    #word = ih[:, :args.we]
    #syn['pos'] = pos.norm(2).item() if parser.pos_in else None
    #syn['word'] = word.norm(2).item()
    #weights['syn'] = syn
    #weights['sem'] = sem
    #sdp_eval_data['weights'] = weights

    save_eval_data(args, experiment['path'], 'sdp', sdp_eval_data)

    print('Finished with syntactic evaluation!')



def eval_sts(args, parser, data, experiment=None):
    device = data['device']

    exp_path = experiment['path']

    parser.eval()

    sem_eval_data = {}

    with open(exp_path, 'a') as exp_file:
        exp_file.write(f'Semantic evaluation for model : {args.model}\n\n')
        with torch.no_grad():
            for year in YEARS:
                curr_data = data['semeval'][year]
                targets = curr_data['targets']
                predictions = []

                pi = parser.pos_in
                
                w1, p1, sl1 = data_utils.prepare_batch_ss([s1 for s1, s2 in curr_data['sent_pairs']])
                w2, p2, sl2 = data_utils.prepare_batch_ss([s2 for s1, s2 in curr_data['sent_pairs']])
                packed_s1, idx_s1, _, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
                packed_s2, idx_s2, _, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
                h1 = unsort(parser.SemanticRNN(packed_s1), idx_s1)
                h2 = unsort(parser.SemanticRNN(packed_s2), idx_s2)

                h1_avg = utils.average_hiddens(h1, sl1.to(device), sum_f_b=args.sum_f_b) 
                h2_avg = utils.average_hiddens(h2, sl2.to(device), sum_f_b=args.sum_f_b) 

                predictions = utils.predict_sts_score(
                        h1_avg,
                        h2_avg,
                        conventional_range=False)

                #predictions = np.random.randn(len(curr_data['targets']))

                correlation = utils.sts_scoring(predictions, targets)

                #scores = [score for score in zip(predictions, targets)]

                #ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

                #plt.legend(loc=2)
                #plt.show()
                #breakpoint()

                #plt.title(f'{year} Task')
                #plt.scatter(predictions, targets, c='blue')
                #plt.xlabel('Predictions')
                #plt.ylabel('Targets')
                #plt.savefig(f'{year}.png')
                #plt.clf()

                info = '{} task\nPearson R*100 {:5.2f}\tAverage gold score:{:5.2f}\tAverage predicted score:{:5.2f}\n'.format(
                    year,
                    100 * correlation,
                    np.mean(targets),
                    np.mean(predictions),
                    )
                stats = 'Std of predictions:{:5.2f}\tStd of targets:{:5.2f}\tCovariance:{:5.2f}\tMin predictions:{:5.2f}\tMax prediction:{:5.2f}\n'.format(
                    np.std(predictions),
                    np.std(targets),
                    np.cov(predictions,targets)[0,1],
                    np.min(predictions),
                    np.max(predictions))
                exp_file.write(info)
                print(info)

                sem_eval_data[year] = correlation
                #print(stats)

    average = 0
    YEARS.remove('2017')
    for year in YEARS:
        average += sem_eval_data[year]
    average /= len(YEARS)
    sem_eval_data['average'] = average
    save_eval_data(args, experiment['path'], 'sem', sem_eval_data)

    print('Finished with semantic evaluation!')


def eval_stag(args, parser, data, experiment=None):
    device = data['device']

    datasets = {'dev': data['dev'], 'test': data['test']}

    stag_eval_data = {}

    with open(experiment['path'], 'a') as exp_file:
        d = experiment['date']
        prelude = '\n' * 3 + f'New supertagging evaluation experiment for model : {args.model}'
        prelude += f'\nStarting date/time: {d:%m %d} at {d:%H:%M:%S}'
        exp_file.write(prelude)

        print(prelude)

        parser.eval()  # Crucial! Toggles dropout effects
        with torch.no_grad():
            for name, dataset in datasets.items():
                loader = data_utils.stag_data_loader(dataset, batch_size=100, shuffle_idx=False)
                n_dev_batches = ceil(len(dataset) / 100)

                total_predictions = 0
                total_correct = 0
                for b in range(n_dev_batches):
                    batch = next(loader)
                    stag_targets = batch['stag_targets']
                    sent_lens = batch['sent_lens'].to(device)
                    pos = batch['pos'].to(device) if parser.pos_in else None

                    lstm_input, indices, lens_sorted, _ = parser.Embeddings(batch['words'].to(device), sent_lens, pos=pos)
                    outputs = parser.SyntacticRNN(lstm_input)
                    logits = parser.StagMLP(unsort(outputs, indices))
                    predictions = torch.argmax(logits, -1)

                    n_correct = torch.eq(predictions, stag_targets.to(device)).sum().item()
                    total_correct += n_correct
                    total_predictions += sent_lens.sum().item()

                stag_accuracy = total_correct / total_predictions

                info = '\nResults for {}:\n {:10.2f}\n'.format(name, 100 * stag_accuracy)
                exp_file.write(info)

                stag_eval_data[name] = stag_accuracy


    save_eval_data(args, experiment['path'], 'stag', stag_eval_data)

    print('Finished with supertagging evaluation!')


def save_eval_data(args, exp_path, task, data):
    #exp_data_path = os.path.splitext(exp_path)[0] + '.pkl'
    exp_data_path = f'../experiments/{args.model}/evaluation.pkl'
    eval_data = {}
    if os.path.exists(exp_data_path):
            with open(exp_data_path, 'rb') as pkl:
                eval_data_pkl = pickle.load(pkl)
            if task in eval_data_pkl:
                eval_data_pkl[task].update(data)
            else:
                eval_data_pkl[task] = data
            eval_data = eval_data_pkl
    else:
        eval_data[task] = data 
    with open(exp_data_path, 'wb') as pkl:
        pickle.dump(eval_data, pkl)


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

