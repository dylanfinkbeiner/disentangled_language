from math import ceil
import logging
import os
import time
import datetime
import pytz
from time import sleep
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from data_utils import sdp_data_loader, idx_loader, prepare_batch_ss, megabatch_breakdown, get_syntactic_scores
from data_utils import prepare_batch_sdp, decode_sdp_sents
import data_utils
import utils 
import losses 
from parser import unsort


LOG_DIR = '../log'
DATA_DIR = '../data'

log = logging.getLogger('__train__')
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'train.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

#MODE_DESC = {
#        -1 : 'dev evaluation',
#        0 : 'normal syntactic parsing',
#        1 : 'just semantic rep training',
#        2 : 'syntactic parsing and semantic rep'
#        3 : 'syntactic rep, semantic rep, and syntactic parsing'
#        }


def train(args, parser, data, weights_path=None, experiment=None):
    # Append to existing training experiment file for given day
    exp_file = open(experiment['path'], 'a')
    d = experiment['date']
    prelude = '\n' * 3 + f'New training experiment for model : {args.model}'
    prelude += f'\nStarting date/time: {d:%m %d} at {d:%H:%M:%S}'
    #description = input('Describe this experiment: ')
    #prelude += f'\nExperiment description: {description}'

    log.info(f'Training model \"{args.model}\" for {args.epochs} epochs in training mode {args.train_mode}.')
    log.info(f'Weights will be saved to {weights_path}.')
    log.info(f'Final hidden size: {args.final_h}, Semantic: {args.sem_h}, Syntactic: {args.syn_h}')
    log.info(f'Scrambling probability: {args.scramble}')

    exp_file.write(prelude)
    sleep(5)

    #torch.manual_seed(args.seed)
    #mode_description = MODE_DESC[args.train_mode]

    train_sdp = data['data_ptb']['train']
    dev_sdp = data['data_ptb']['dev']
    loader_sdp_train = sdp_data_loader(train_sdp, batch_size=args.batch_size, shuffle_idx=True)
    loader_sdp_dev = sdp_data_loader(dev_sdp, batch_size=100, shuffle_idx=False)
    log.info(f'There are {len(train_sdp)} syntactic dependency parsing training examples.')
    log.info(f'There are {len(dev_sdp)} syntactic dependency parsing dev examples.')
    if args.train_mode > 0:
        train_ss = data['data_ss']['train']['sent_pairs']
        dev_ss = data['data_ss']['dev']
        idxloader_ss_train = idx_loader(num_data=len(train_ss), batch_size=args.batch_size)
        log.info(f'There are {len(train_ss)} semantic similarity training examples.')
        log.info(f"There are {len(dev_ss['sent_pairs'])} semantic similarity dev examples.")
    if args.train_mode > 3:
        train_stag = data['data_stag']['train']
        dev_stag = data['data_stag']['dev']
        loader_stag_train = stag_data_loader(train_stag, batch_size=args.batch_size, shuffle_idx=True)
        loader_stag_dev = stag_data_loader(dev_stag, batch_size=100, shuffle_idx=False)
        log.info(f'There are {len(train_stag)} supertagging training examples.')
        log.info(f'There are {len(dev_stag)} supertagging dev examples.')


    n_train_batches = ceil(len(train_sdp) / args.batch_size)
    n_dev_batches = ceil(len(dev_sdp) / 100)
    if args.train_mode > 0:
        n_megabatches = ceil(len(train_ss) / (args.M * args.batch_size))

    opt = Adam(parser.parameters(), lr=2e-3, betas=[0.9, 0.9])

    exp_file.write('Training results:')
    earlystop_counter = 0
    prev_best = 0
    #if not args.init_model and args.train_mode != -1:
    _, prev_best, _ = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)

    print_grad_every = 10
    try:
        for e in range(args.epochs):
            log.info(f'Entering epoch {e+1}/{args.epochs}.')

            train_loss = 0
            num_steps = 0
            if args.train_mode == 0:
                for b in tqdm(range(n_train_batches), ascii=True, desc=f'Epoch {e+1}/{args.epochs} progress', ncols=80):
                    opt.zero_grad()
                    batch = next(loader_sdp_train)
                    loss = forward_syntactic_parsing(parser, batch, args=args, data=data)
                    loss.backward()
                    opt.step()

            elif args.train_mode > 0:
                for m in range(n_megabatches):
                    megabatch = []
                    idxs = []
                    idx = 0
                    for _ in range(args.M):
                        batch = [train_ss[j] for j in next(idxloader_ss_train)]
                        curr_idxs = [i + idx for i in range(len(batch))]
                        megabatch.extend(batch)
                        idxs.append(curr_idxs)
                        idx += len(curr_idxs)

                    with torch.no_grad():
                        mb_para1, mb_para2, mb_neg1, mb_neg2 = megabatch_breakdown(
                                megabatch, 
                                minibatch_size=args.batch_size, 
                                parser=parser,
                                args=args,
                                data=data)

                    for x in tqdm(range(len(idxs)), ascii=True, desc=f'Megabatch {m+1}/{n_megabatches} progress, e:{e+1}', ncols=80):
                        loss = None
                        opt.zero_grad()
                        if args.train_mode == 2:
                            #Standard syntactic parsing step
                            batch = next(loader_sdp_train)
                            loss = forward_syntactic_parsing(parser, batch, args=args, data=data)

                        if args.train_mode == 3:
                            batch = next(loader_sdp_train)
                            loss_par = forward_syntactic_parsing(
                                    parser, 
                                    batch=batch, 
                                    args=args, 
                                    data=data)
                            loss_pos = forward_pos(
                                    parser,
                                    batch=batch,
                                    args=args,
                                    data=data)
                            loss = loss_par + loss_pos

                        if args.train_mode == 4:
                            sdp_batch = next(loader_sdp_train)
                            loss_par = forward_syntactic_parsing(
                                    parser, 
                                    batch=sdp_batch, 
                                    args=args, 
                                    data=data)
                            stag_batch = next(loader_stag_train)
                            loss_pos = forward_stag(
                                    parser,
                                    batch=stag_batch,
                                    args=args,
                                    data=data)
                            loss = loss_par + loss_stag

                        # Sentence similarity step
                        loss_sem = forward_semantic(
                                parser,
                                para1=[mb_para1[i] for i in idxs[x]],
                                para2=[mb_para2[i] for i in idxs[x]],
                                neg1=[mb_neg1[i] for i in idxs[x]],
                                neg2=[mb_neg2[i] for i in idxs[x]] if mb_neg2 is not None else None,
                                args=args,
                                data=data)
                        loss += loss_sem.cpu()
                        loss.backward()
                        if x % print_grad_every == 0:
                            print(f'\nGradient:\n')
                            print(gradient_update(parser, verbose=False))
                        
                        opt.step()

                    update = f'''\nUpdate for megabatch: {m+1}\n'''
                    correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                    update += '''Semantic train loss: {:.4f}
                            Semantic dev loss: {:.4f}
                            Correlation: {:.4f}'''.format(-2, -2, correlation)

                    log.info(update)

            update = f'''\nUpdate for epoch: {e+1}/{args.epochs}\n'''
            #if args.train_mode  2:
            if True:
                UAS, LAS, dev_loss = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)
                update += '''\n
                        Syntactic train loss: {:.3f}
                        Syntactic dev loss: {:.3f}
                        UAS * 100: {:.3f}
                        LAS * 100: {:.3f}'''.format(train_loss, dev_loss, UAS * 100, LAS * 100)
            if args.train_mode > 0:
                correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                update += '''\n
                        Semantic train loss: {:.4f}
                        Semantic dev loss: {:.4f}
                        Correlation: {:.4f}'''.format(-2, -2, correlation)

            log.info(update)
            exp_file.write(update)

            # Early stopping heuristic from Jabberwocky paper
            #if args.train_mode != 2:
            if True:
                if LAS > prev_best:
                    print(f'LAS improved from {prev_best} on epoch {e+1}/{args.epochs}.')
                    earlystop_counter = 0
                    prev_best = LAS
                else:
                    earlystop_counter += 1
                    print(f'LAS has not improved for {earlystop_counter} consecutive epoch(s).')
                    if earlystop_counter >= 3:
                        print(f'Stopping after {e+1} epochs')
                        break
            
            if args.train_mode != -1:
                torch.save(parser.state_dict(), weights_path)
                log.info(f'Weights saved to {weights_path}.')

    except KeyboardInterrupt:
        response = input("Keyboard interruption: Would you like to save weights? [y/n]")
        if response.lower() == 'y':
            torch.save(parser.state_dict(), weights_path)
        exp_file.write(f'\n\nExperiment halted by keyboard interrupt. Weights saved : {response.lower()}')

    finally:
        d = datetime.datetime.utcnow()
        d = d.astimezone(pytz.timezone("America/Los_Angeles"))
        exp_file.write('\n'*3 + f'Experiment ended at {d:%H:%M:%S}')
        exp_file.close()


def forward_syntactic_parsing(parser, batch, args=None, data=None):
    device = data['device']
    parser.train()

    arc_targets = batch['arc_targets']
    rel_targets = batch['rel_targets']
    sent_lens = batch['sent_lens'].to(device)
    
    words_d = utils.word_dropout(
            batch['words'], 
            w2i=data['vocabs']['x2i']['word'], 
            i2w=data['vocabs']['i2x']['word'], 
            counts=data['word_counts'], 
            lens=sent_lens,
            alpha=args.alpha)

    S_arc, S_rel, _ = parser(words_d.to(device), sent_lens, pos=batch['pos'].to(device)) #XXX
    #S_arc, S_rel, _ = parser(words_d.to(device), sent_lens)
    
    loss_h = losses.loss_arcs(S_arc, arc_targets)
    loss_r = losses.loss_rels(S_rel, rel_targets)
    loss = loss_h + loss_r
    
    #loss *= args.lr_syn

    return loss


def forward_pos(parser, batch, args=None, data=None):
    device = data['device']
    parser.train()

    arc_targets = batch['arc_targets']
    rel_targets = batch['rel_targets']
    pos_targets = batch['pos'].to(device)
    sent_lens = batch['sent_lens'].to(device)
    
    lstm_input, indices, lens_sorted = parser.Embeddings(batch['words'].to(device), sent_lens)
    #lstm_input, indices, lens_sorted = parser.Embeddings(batch['words'].to(device), sent_lens, pos=batch['pos'].to(device))
    outputs = parser.SyntacticRNN(lstm_input)
    logits = parser.POSMLP(unsort(outputs, indices))

    loss_pos = losses.loss_pos(logits, pos_targets).cpu()
    
    #loss *= args.lr_syn

    return loss_pos


def forward_semantic(parser, para1, para2, neg1, neg2=None, args=None, data=None):
    device = data['device']
    parser.train()

    #w1, p1, sl1 = prepare_batch_ss(para1)
    #w2, p2, sl2 = prepare_batch_ss(para2)
    #wn1, pn1, sln1 = prepare_batch_ss(neg1)
    w1, _, sl1 = prepare_batch_ss(para1)
    w2, _, sl2 = prepare_batch_ss(para2)
    wn1, _, sln1 = prepare_batch_ss(neg1)
    
    #packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1, p1.to(device))
    #packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device),sl2,  p2.to(device))
    #packed_n1, idx_n1, _ = parser.Embeddings(wn1.to(device), sln1, pn1.to(device))
    packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1)
    packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device), sl2)
    packed_n1, idx_n1, _ = parser.Embeddings(wn1.to(device), sln1)

    h1 = unsort(parser.SemanticRNN(packed_s1), idx_s1)
    h2 = unsort(parser.SemanticRNN(packed_s2), idx_s2)
    hn1 = unsort(parser.SemanticRNN(packed_n1), idx_n1)

    h1_avg = utils.average_hiddens(h1, sl1.to(device), sum_f_b=args.sum_f_b)
    h2_avg = utils.average_hiddens(h2, sl2.to(device), sum_f_b=args.sum_f_b)
    hn1_avg = utils.average_hiddens(hn1, sln1.to(device), sum_f_b=args.sum_f_b)

    if neg2 is not None:
        wn2, pn2, sln2 = prepare_batch_ss(neg2)
        #wn2, _, sln2 = prepare_batch_ss(neg2)
        packed_n2, idx_n2 = parser.Embeddings(wn2.to(device), pn2.to(device), sln2.to(device))
        #packed_n2, idx_n2 = parser.Embeddings(wn2.to(device), sln2.to(device))
        hn2 = unsort(parser.SemanticRNN(packed_n2), idx_n2)
        hn2_avg = utils.average_hiddens(hn2, sln2.to(device), sum_f_b=args.sum_f_b)
    else:
        hn2_avg = None
    
    loss = losses.loss_sem_rep(
            h1_avg,
            h2_avg,
            hn1_avg,
            sem_hn2=hn2_avg,
            margin=args.margin)

    #loss *= args.lr_sem

    return loss


def sdp_dev_eval(parser, args=None, data=None, loader=None, n_dev_batches=None):
    device = data['device']

    parser.eval()  # Crucial! Toggles dropout effects
    dev_loss = 0
    UAS = 0
    LAS = 0
    total_words = 0
    log.info('Evaluation step begins.')
    with torch.no_grad():
        for b in range(n_dev_batches):
            batch = next(loader)
            arc_targets = batch['arc_targets']
            rel_targets = batch['rel_targets']
            sent_lens = batch['sent_lens'].to(device)
    
            S_arc, S_rel, arc_preds = parser(
                    batch['words'].to(device), 
                    #sent_lens)
                    sent_lens, #XXX
                    pos=batch['pos'].to(device)) # XXX
            rel_preds = utils.predict_relations(S_rel)
    
            loss_h = losses.loss_arcs(S_arc, arc_targets)
            loss_r = losses.loss_rels(S_rel, rel_targets)
            dev_loss += loss_h.item() + loss_r.item()
    
            results = utils.attachment_scoring(
                    arc_preds=arc_preds.cpu(),
                    #arc_preds=arc_preds,
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


def ss_dev_eval(parser, dev_ss, args=None, data=None):
    device = data['device']

    parser.eval()
    correlation = -1337.0
    with torch.no_grad():
        #w1, p1, sl1 = prepare_batch_ss([s1 for s1, _ in dev_ss['sent_pairs']])
        #w2, p2, sl2 = prepare_batch_ss([s2 for _, s2 in dev_ss['sent_pairs']])
        w1, _, sl1 = prepare_batch_ss([s1 for s1, _ in dev_ss['sent_pairs']])
        w2, _, sl2 = prepare_batch_ss([s2 for _, s2 in dev_ss['sent_pairs']])

        #packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1, p1.to(device))
        #packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device), sl2, p2.to(device))
        packed_s1, idx_s1, _ = parser.Embeddings(w1.to(device), sl1)
        packed_s2, idx_s2, _ = parser.Embeddings(w2.to(device), sl2)
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
