from math import ceil
import logging
import os
import time
import datetime as dt
import pytz
from time import sleep
from tqdm import tqdm
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import data_utils
import utils 
import losses 
from parser import unsort

LOG_DIR = '../log'
DATA_DIR = '../data'




def write_prelude(args, experiment):
    d = experiment['date']
    prelude = []
    prelude.append('\n' * 3 + f'New training experiment for model : {args.model}')
    prelude.append(f'Starting date/time: {d:%m %d} at {d:%H:%M:%S}')
    #description = input('Describe this experiment: ')
    #prelude += f'\nExperiment description: {description}'
    prelude.append(f'Training model \"{args.model}\" for {args.epochs} epochs in training mode {args.train_mode}.')
    prelude.append(f'Final hidden size: {args.final_h}, Semantic: {args.sem_h}, Syntactic: {args.syn_h}')
    prelude.append(f'Scrambling probability: {args.scramble}')
    prelude.append('\n' * 3)
    return '\n'.join(prelude)


def get_logger(experiment):
    log = logging.getLogger('__train__')
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    log.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(experiment['path'])
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    return log


def train(args, parser, data, weights_path=None, experiment=None):
    all_to_gpu = True #XXX

    # Append to existing training experiment file for given day
    prelude = write_prelude(args, experiment)
    log = get_logger(experiment)
    log.info(prelude)
    sleep(5)

    train_sdp = data['data_ptb']['train']
    dev_sdp = data['data_ptb']['dev']
    loader_sdp_train = data_utils.sdp_data_loader(train_sdp, batch_size=args.sdp_bs, shuffle_idx=True)
    loader_sdp_dev = data_utils.sdp_data_loader(dev_sdp, batch_size=100, shuffle_idx=False)
    log.info(f'There are {len(train_sdp)} syntactic dependency parsing training examples.')
    log.info(f'There are {len(dev_sdp)} syntactic dependency parsing dev examples.')
    sdp_devloss_record = []
    sdp_uas_record = []
    sdp_las_record = []
    if args.train_mode > 0:
        train_ss = data['data_ss']['train']['sent_pairs']
        dev_ss = data['data_ss']['dev']
        idxloader_ss_train = data_utils.idx_loader(num_data=len(train_ss), batch_size=args.sem_bs)
        log.info(f'There are {len(train_ss)} semantic similarity training examples.')
        log.info(f"There are {len(dev_ss['sent_pairs'])} semantic similarity dev examples.")
        sem_corr_record = []
    if args.train_mode > 3:
        train_stag = data['data_stag']['train']
        dev_stag = data['data_stag']['dev']
        loader_stag_train = data_utils.stag_data_loader(train_stag, batch_size=args.stag_bs, shuffle_idx=True)
        loader_stag_dev = data_utils.stag_data_loader(dev_stag, batch_size=100, shuffle_idx=False)
        log.info(f'There are {len(train_stag)} supertagging training examples.')
        log.info(f'There are {len(dev_stag)} supertagging dev examples.')
        stag_acc_record = []


    n_train_batches = ceil(len(train_sdp) / args.sdp_bs)
    n_dev_batches = ceil(len(dev_sdp) / 100)
    if args.train_mode > 0:
        n_megabatches = ceil(len(train_ss) / (args.M * args.sem_bs))

    opt = Adam(parser.parameters(), lr=args.lr, betas=[0.9, 0.9])

    print_grad_every = 10
    earlystop_counter = 0
    _, prev_best, _ = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)
    try:
        for e in range(args.epochs):
            log.info(f'Entering epoch {e+1}/{args.epochs}.')

            if args.train_mode == 0:
                for b in tqdm(range(n_train_batches), ascii=True, desc=f'Epoch {e+1}/{args.epochs} progress', ncols=80):
                    opt.zero_grad()
                    batch = next(loader_sdp_train)
                    loss = forward_syntactic_parsing(parser, batch, args=args, data=data)
                    loss.backward()
                    opt.step()

            elif args.train_mode > 0:
                mini_remaining = ceil(len(train_ss) / args.sem_bs)
                for m in range(n_megabatches):
                    megabatch = []
                    idxs = [] # List of lists (minis of indices into flat mega)
                    idx = 0
                    for _ in range(min(args.M, mini_remaining)):
                        batch = [train_ss[i] for i in next(idxloader_ss_train)]
                        mini_remaining -= 1
                        curr_idxs = [(i + idx) for i in range(len(batch))]
                        megabatch.extend(batch)
                        idxs.append(curr_idxs)
                        idx += len(curr_idxs)

                    with torch.no_grad():
                        mb_s1, mb_s2, mb_n1, mb_n2 = data_utils.megabatch_breakdown(
                                megabatch, 
                                minibatch_size=args.sem_bs,
                                parser=parser,
                                args=args,
                                data=data)

                    for x in tqdm(range(len(idxs)), ascii=True, desc=f'Megabatch {m+1}/{n_megabatches} progress, e:{e+1}', ncols=80):
                        loss = None
                        opt.zero_grad()

                        if args.train_mode == 2:
                            batch = next(loader_sdp_train)
                            loss = forward_syntactic_parsing(
                                    parser, 
                                    batch,
                                    args=args, 
                                    data=data)

                        if args.train_mode == 4:
                            sdp_batch = next(loader_sdp_train)
                            loss_sdp = forward_syntactic_parsing(
                                    parser, 
                                    batch=sdp_batch, 
                                    args=args, 
                                    data=data)
                            stag_batch = next(loader_stag_train)
                            loss_stag = forward_stag(
                                    parser,
                                    batch=stag_batch,
                                    args=args,
                                    data=data)
                            loss = loss_sdp + loss_stag

                        loss_sem = forward_semantic_similarity(
                                parser,
                                s1=[mb_s1[i] for i in idxs[x]],
                                s2=[mb_s2[i] for i in idxs[x]],
                                n1=[mb_n1[i] for i in idxs[x]],
                                n2=[mb_n2[i] for i in idxs[x]] if mb_n2 is not None else None,
                                args=args,
                                data=data)
                        loss += loss_sem

                        loss.backward()
                        opt.step()

                        if x % print_grad_every == 0:
                            print(gradient_update(parser, verbose=False))

                    update = ''
                    #update += f'''\nUpdate for megabatch: {m+1}\n'''
                    #correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                    #update += '''Correlation: {:.4f}'''.format(correlation)

                    log.info(update)
                    

            update = f'''\nUpdate for epoch: {e+1}/{args.epochs}\n'''
            UAS, LAS, dev_loss = sdp_dev_eval(
                    parser, 
                    args=args, 
                    data=data, 
                    loader=loader_sdp_dev,
                    n_dev_batches=n_dev_batches)
            sdp_devloss_record.append(dev_loss)
            sdp_uas_record.append(UAS)
            sdp_las_record.append(LAS)
            update += '''\nSyntactic dev loss: {:.3f}
                    UAS * 100: {:.3f}
                    LAS * 100: {:.3f}\n'''.format(dev_loss, UAS * 100, LAS * 100)
            if args.train_mode > 0:
                correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                sem_corr_record.append(correlation)
                update += '''\nCorrelation: {:.4f}\n'''.format(correlation)
            if args.train_mode > 3:
                stag_accuracy = stag_dev_eval(
                        parser, 
                        args=args, 
                        data=data, 
                        loader=loader_stag_dev, 
                        n_dev_batches=ceil(len(dev_stag) / 100))
                stag_acc_record.append(stag_accuracy)
                update += '''\nSupertagging accuracy: {:.4f}\n'''.format(stag_accuracy)
            log.info(update)

            if LAS > prev_best:
                msg = f'LAS improved from {prev_best} on epoch {e+1}/{args.epochs}.'
                log.info(msg)
                earlystop_counter = 0
                prev_best = LAS
            else:
                earlystop_counter += 1
                msg = f'LAS has not improved for {earlystop_counter} consecutive epoch(s).'
                log.info(msg)
                if earlystop_counter >= 2:
                    break
            
            if args.train_mode != -1:
                torch.save(parser.state_dict(), weights_path)
                log.info(f'Weights saved to {weights_path}.')


    except KeyboardInterrupt:
        if not args.autopilot:
            response = input("Keyboard interruption: Would you like to save weights? [y/n]")
            if response.lower() == 'y':
                torch.save(parser.state_dict(), weights_path)
                log.info(f'\n\nExperiment halted by keyboard interrupt. Weights saved : {response.lower()}')
        else:
            torch.save(parser.state_dict(), weights_path)

    finally:
        with open(os.path.splitext(experiment['path'])[0] + '.pkl', 'wb') as pkl:
                exp_data = {}
                exp_data['sdp_devloss_record'] = sdp_devloss_record
                exp_data['sdp_uas_record'] = sdp_uas_record
                exp_data['sdp_las_record'] = sdp_las_record
                exp_data['sem_corr_record'] = sem_corr_record
                exp_data['stag_acc_record'] = stag_acc_record
                pickle.dump(exp_data, pkl)

        d = dt.datetime.now().astimezone(pytz.timezone("America/Los_Angeles"))
        log.info('\n'*3 + f'Experiment ended at {d:%H:%M:%S}\n')


def forward_syntactic_parsing(parser, batch, args=None, data=None):
    parser.train()
    device = data['device']
    sent_lens = batch['sent_lens']
    pos = batch['pos'].to(device) if parser.pos_in else None
    
    words_d = utils.word_dropout(
            batch['words'], 
            w2i=data['vocabs']['x2i']['word'], 
            i2w=data['vocabs']['i2x']['word'], 
            counts=data['word_counts'], 
            lens=sent_lens,
            alpha=args.alpha)

    S_arc, S_rel, _ = parser(words_d.to(device), sent_lens, pos=pos)
    
    loss_a = losses.loss_arcs(S_arc, batch['arc_targets'].to(device))
    loss_r = losses.loss_rels(S_rel, batch['rel_targets'].to(device))
    loss = loss_a + loss_r
    
    loss *= args.lr_sdp

    return loss


def forward_stag(parser, batch, args=None, data=None):
    parser.train()
    device = data['device']

    stag_targets = batch['stag_targets'].to(device)
    sent_lens = batch['sent_lens']
    pos = batch['pos'].to(device) if parser.pos_in else None

    lstm_input, indices, _ = parser.Embeddings(
            batch['words'].to(device), 
            sent_lens, 
            pos=pos)
    outputs = parser.SyntacticRNN(lstm_input)
    logits = parser.StagMLP(unsort(outputs, indices))

    loss = losses.loss_stag(logits, stag_targets)
    
    loss *= args.lr_stag

    return loss


def forward_semantic_similarity(parser, s1, s2, n1, n2=None, args=None, data=None):
    parser.train()
    device = data['device']

    w1, p1, sl1 = data_utils.prepare_batch_ss(s1)
    w2, p2, sl2 = data_utils.prepare_batch_ss(s2)
    wn1, pn1, sln1 = data_utils.prepare_batch_ss(n1)

    pi = parser.pos_in
    
    packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
    packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
    packed_n1, idx_n1, _ = parser.Embeddings(wn1.to(device), sln1, pos=pn1.to(device) if pi else None)

    h1 = unsort(parser.SemanticRNN(packed_s1), idx_s1)
    h2 = unsort(parser.SemanticRNN(packed_s2), idx_s2)
    hn1 = unsort(parser.SemanticRNN(packed_n1), idx_n1)

    h1_avg = utils.average_hiddens(h1, sl1.to(device), sum_f_b=args.sum_f_b)
    h2_avg = utils.average_hiddens(h2, sl2.to(device), sum_f_b=args.sum_f_b)
    hn1_avg = utils.average_hiddens(hn1, sln1.to(device), sum_f_b=args.sum_f_b)

    hn2_avg = None
    if n2 is not None:
        wn2, pn2, sln2 = data_utils.prepare_batch_ss(n2)
        packed_n2, idx_n2 = parser.Embeddings(wn2.to(device), sln2.to(device), pos=pn2.to(device) if pi else None)
        hn2 = unsort(parser.SemanticRNN(packed_n2), idx_n2)
        hn2_avg = utils.average_hiddens(hn2, sln2.to(device), sum_f_b=args.sum_f_b)
    
    loss = losses.loss_sem_rep(
            h1_avg,
            h2_avg,
            hn1_avg,
            sem_hn2=hn2_avg,
            margin=args.margin)

    loss *= args.lr_sem

    return loss


def sdp_dev_eval(parser, args=None, data=None, loader=None, n_dev_batches=None):
    device = data['device']

    parser.eval()
    dev_loss = 0
    UAS = 0
    LAS = 0
    total_words = 0
    with torch.no_grad():
        for b in range(n_dev_batches):
            batch = next(loader)
            arc_targets = batch['arc_targets'].to(device)
            rel_targets = batch['rel_targets'].to(device)
            sent_lens = batch['sent_lens']
            pos = batch['pos'].to(device) if parser.pos_in else None
    
            S_arc, S_rel, arc_preds = parser(
                    batch['words'].to(device), 
                    sent_lens,
                    pos=pos)
            rel_preds = utils.predict_relations(S_rel)
    
            loss_a = losses.loss_arcs(S_arc, arc_targets)
            loss_r = losses.loss_rels(S_rel, rel_targets)
            dev_loss += loss_a.item() + loss_r.item()
    
            results = utils.attachment_scoring(
                    arc_preds=arc_preds.cpu(),
                    rel_preds=rel_preds,
                    arc_targets=arc_targets,
                    rel_targets=rel_targets,
                    sent_lens=sent_lens,
                    include_root=False)
            UAS += results['UAS_correct']
            LAS += results['LAS_correct']
            total_words += results['total_words']

    dev_loss /= n_dev_batches
    UAS /= total_words
    LAS /= total_words

    return UAS, LAS, dev_loss


def stag_dev_eval(parser, args=None, data=None, loader=None, n_dev_batches=None):
    device = data['device']

    parser.eval()
    total_predictions = 0
    total_correct = 0
    with torch.no_grad():
        for b in range(n_dev_batches):
            batch = next(loader)
            stag_targets = batch['stag_targets']
            sent_lens = batch['sent_lens'].to(device)
            pos = batch['pos'].to(device) if parser.pos_in else None

            lstm_input, indices, _ = parser.Embeddings(
                    batch['words'].to(device), 
                    sent_lens, 
                    pos=pos)
            outputs = parser.SyntacticRNN(lstm_input)
            logits = parser.StagMLP(unsort(outputs, indices))
            predictions = torch.argmax(logits, -1)

            n_correct = torch.eq(predictions, stag_targets.to(device)).sum().item()
            total_correct += n_correct
            total_predictions += sent_lens.sum().item()
   
    stag_accuracy = total_correct / total_predictions

    return stag_accuracy


def ss_dev_eval(parser, dev_ss, args=None, data=None):
    device = data['device']

    pi = parser.pos_in

    parser.eval()
    correlation = -1337.0
    with torch.no_grad():
        w1, p1, sl1 = data_utils.prepare_batch_ss([s1 for s1, _ in dev_ss['sent_pairs']])
        w2, p2, sl2 = data_utils.prepare_batch_ss([s2 for _, s2 in dev_ss['sent_pairs']])

        packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
        packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
        h1 = unsort(parser.SemanticRNN(packed_s1), idx_s1)
        h2 = unsort(parser.SemanticRNN(packed_s2), idx_s2)

        predictions = utils.predict_sts_score(
                utils.average_hiddens(h1, sl1.to(device), sum_f_b=args.sum_f_b), 
                utils.average_hiddens(h2, sl2.to(device), sum_f_b=args.sum_f_b))
        
        correlation = utils.sts_scoring(predictions, dev_ss['targets'])

    return correlation


def gradient_update(parser, verbose=False):
    update = ''
    t = 0
    for p in parser.parameters():
        if type(p.grad) != type(None):
            n = p.grad.data.norm(2)
            t += n.item() ** 2
    t = t ** (1. / 2)
    if verbose:
        update += '\n'.join(['{:35} {:3.8f}'.format(n, p.grad.data.norm(2).item()) for n, p in list(parser.named_parameters())])
        update += f'\nTotal norm of gradient is {t}\n'
    else: 
        update += 'Total norm of gradient: {:5.8f}\n'.format(t)

    return update
