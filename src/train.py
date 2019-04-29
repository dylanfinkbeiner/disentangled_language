from math import ceil
import logging
import os
import time
from time import sleep
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from data_utils import sdp_data_loader_original, idx_loader, prepare_batch_ss, megabatch_breakdown, get_syntactic_scores
from data_utils import prepare_batch_sdp, decode_sdp_sents
import data_utils
from losses import loss_arcs, loss_rels, loss_sem_rep, loss_syn_rep
import utils 


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
#        1 : 'custom task'
#        }


def train(args, parser, data, weights_path=None, exp_path_base=None):
    exp_path = '_'.join([exp_path_base, str(args.train_mode)])
    exp_file = open(exp_path, 'a')
    exp_file.write(f'Training experiment for model : {args.model}')

    description = input('Describe this experiment: ')
    exp_file.write('\n\nExperiment description: {description} \n\n')

    log.info(f'Training model \"{args.model}\" for {args.epochs} epochs in training mode {args.train_mode}.')
    log.info(f'Weights will be saved to {weights_path}.')
    sleep(5)


    torch.manual_seed(args.seed)
    #mode_description = MODE_DESC[args.train_mode]

    train_sdp = data['data_ptb']['train']
    #data_utils.sdp_corpus_stats(train_sdp, device=data['device'])
    #exit()

    dev_sdp = data['data_ptb']['dev']
    loader_sdp_train = sdp_data_loader_original(train_sdp, batch_size=args.batch_size, shuffle_idx=True, custom_task=True)
    loader_sdp_dev = sdp_data_loader_original(dev_sdp, batch_size=100, shuffle_idx=False, custom_task=False)
    log.info(f'There are {len(train_sdp)} SDP training examples.')
    log.info(f'There are {len(dev_sdp)} SDP dev examples.')
    if args.train_mode > 0:
        train_ss = data['data_ss']['train']['sent_pairs']
        dev_ss = data['data_ss']['dev']
        idxloader_ss_train = idx_loader(num_data=len(train_ss), batch_size=args.batch_size)
        log.info(f'There are {len(train_ss)} SS training examples.')
        log.info(f"There are {len(dev_ss['sent_pairs'])} SS dev examples.")

    n_train_batches = ceil(len(train_sdp) / args.batch_size)
    n_dev_batches = ceil(len(dev_sdp) / 100)
    if args.train_mode > 0:
        n_megabatches = ceil(len(train_ss) / (args.M * args.batch_size))

    opt = Adam(parser.parameters(), lr=1.0, betas=[0.9, 0.9])


    exp_file.write('Training results:')
    earlystop_counter = 0
    prev_best = 0
    if not args.init_model and args.train_mode != -1:
        _, prev_best, _ = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)

    print_grad_every = 10
    try:
        for e in range(args.epochs):
            log.info(f'Entering epoch {e+1}/{args.epochs}.')

            #parser.train()
            train_loss = 0
            num_steps = 0
            #if mode_description == 'normal syntactic parsing': 
            if args.train_mode == 0:
                for b in tqdm(range(n_train_batches), ascii=True, desc=f'Epoch {e} progress', ncols=80):
                    opt.zero_grad()
                    batch = next(loader_sdp_train)
                    loss = forward_syntactic_original(parser, batch, args=args, data=data)
                    loss.backward()
                    opt.step()
                    #num_steps += 1

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
                        mb_para1, mb_para2, mb_neg1 = megabatch_breakdown(megabatch, args.batch_size, parser, data['device'])

                    for x in tqdm(range(len(idxs)), ascii=True, desc=f'Megabatch {m+1}/{n_megabatches} progress', ncols=80):
                        #log.info(f'Epoch {e+1}/{args.epochs}, megabatch {m+1}/{n_megabatches}, batch {x+1}/{len(idxs)}')

                        if args.train_mode != 2:
                            # Syntactic step
                            opt.zero_grad()

                            #sample = train_sdp[100:103]
                            #decoded = decode_sdp_sents(sents=sample, i2x=data['vocabs']['i2x'])

                            s1_batch, s2_batch, scores_batch = next(loader_sdp_train)

                            loss_syn = forward_syntactic_custom(parser,
                                    s1=s1_batch, 
                                    s2=s2_batch, 
                                    scores=scores_batch, args=args, data=data)

                            #train_loss += loss.item()
                            #num_steps += 1
                            loss_syn.backward()
                            opt.step()

                            if x % print_grad_every == 0:
                                update = gradient_update(parser)
                                log.info(update)
                                exp_file.write(update)

                        # Sentence similarity step
                        opt.zero_grad()
                        loss_sem = forward_semantic(parser,
                                para1=[mb_para1[i] for i in idxs[x]],
                                para2=[mb_para2[i] for i in idxs[x]],
                                neg1=[mb_neg1[i] for i in idxs[x]],
                                args=args,
                                data=data)
                        loss_sem.backward()
                        opt.step()

            #train_loss /= (num_steps if num_steps > 0 else -1) # Just dependency parsing loss

            update = f'''Update for epoch: {e+1/args.epochs}\n'''
            if args.train_mode != 2:
                UAS, LAS, dev_loss = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)
                update += '''\nSyntactic train loss: {:.3f}
                        Syntactic dev loss: {:.3f}
                        UAS: {:.3f}
                        LAS: {:.3f}'''.format(train_loss, dev_loss, UAS * 100, LAS * 100)
            if args.train_mode > 0:
                correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                update += '''\nSemantic train loss: {:.4f}
                        Semantic dev loss: {:.4f}
                        Correlation: {:.4f}'''.format(69, 420, correlation)

            log.info(update)
            exp_file.write(update)

            # Early stopping heuristic from Jabberwocky paper
            if args.train_mode != 2:
                if LAS > prev_best:
                    print(f'LAS improved from {prev_best} on epoch {e+1/args.epochs}.')
                    earlystop_counter = 0
                    prev_best = LAS
                else:
                    earlystop_counter += 1
                    print(f'LAS has not improved for {earlystop_counter} consecutive epoch(s).')
                    if earlystop_counter >= 5:
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
        exp_file.close()


def forward_syntactic_original(parser, batch, args=None, data=None):
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
            lens=sent_lens)
    
    S_arc, S_rel, _ = parser(words_d.to(device), batch['pos'].to(device), sent_lens)
    
    loss_h = loss_arcs(S_arc, arc_targets)
    loss_r = loss_rels(S_rel, rel_targets)
    loss = loss_h + loss_r
    
    loss *= args.lr_syn

    return loss


def forward_syntactic_custom(parser, s1, s2, scores, args=None, data=None):
    device = data['device']

    parser.train()
    w2i = data['vocabs']['x2i']['word'] 
    i2w = data['vocabs']['i2x']['word']

    s1_lens = s1['sent_lens'].to(device)
    s2_lens = s2['sent_lens'].to(device)
    
    words_d_s1 = utils.word_dropout(s1['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=s1_lens)
    words_d_s2 = utils.word_dropout(s2['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=s2_lens)
    
    h_s1, _ = parser.BiLSTM(words_d_s1.to(device), s1['pos'].to(device), s1_lens)
    h_s2, _ = parser.BiLSTM(words_d_s2.to(device), s2['pos'].to(device), s2_lens)
    
    S_arc_s1, S_rel_s1, _ = parser.BiAffineAttention(h_s1.to(device), s1_lens)
    S_arc_s2, S_rel_s2, _ = parser.BiAffineAttention(h_s2.to(device), s2_lens)
    
    loss_h_s1 = loss_arcs(S_arc_s1, s1['arc_targets'])
    loss_r_s1 = loss_rels(S_rel_s1, s1['rel_targets'])
    loss_s1 = loss_h_s1 + loss_r_s1
    loss_h_s2 = loss_arcs(S_arc_s2, s2['arc_targets'])
    loss_r_s2 = loss_rels(S_rel_s2, s2['rel_targets'])
    loss_s2 = loss_h_s2 + loss_r_s2
    
    loss_rep = loss_syn_rep(
            utils.average_hiddens(h_s1, s1_lens),
            utils.average_hiddens(h_s2, s2_lens),
            scores_batch.to(device), 
            syn_size=args.syn_size,
            h_size=args.h_size)
    
    loss = loss_s1.to(device) + loss_s2.to(device) + loss_rep
    loss *= args.lr_syn

    return loss


def forward_semantic(parser, para1, para2, neg1, args=None, data=None):
    device = data['device']

    parser.train()

    w1, p1, sl1 = prepare_batch_ss(para1)
    w2, p2, sl2 = prepare_batch_ss(para2)
    wn, pn, sln = prepare_batch_ss(neg1)
    
    h1, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1.to(device))
    h2, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2.to(device))
    hn, _ = parser.BiLSTM(wn.to(device), pn.to(device), sln.to(device))
    
    loss = loss_sem_rep(
            utils.average_hiddens(h1, sl1.to(device)),
            utils.average_hiddens(h2, sl2.to(device)),
            utils.average_hiddens(hn, sln.to(device)),
            margin=args.margin,
            syn_size=args.syn_size,
            h_size=args.h_size)
    
    loss *= args.lr_sem

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
                    batch['pos'].to(device), 
                    sent_lens)
            rel_preds = utils.predict_relations(S_rel)
    
            loss_h = loss_arcs(S_arc, arc_targets)
            loss_r = loss_rels(S_rel, rel_targets)
            dev_loss += loss_h.item() + loss_r.item()
    
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


def ss_dev_eval(parser, dev_ss, args=None, data=None):
    device = data['device']

    parser.eval()
    correlation = -1337.0
    with torch.no_grad():
        w1, p1, sl1 = prepare_batch_ss([s1 for s1, _ in dev_ss['sent_pairs']])
        w2, p2, sl2 = prepare_batch_ss([s2 for _, s2 in dev_ss['sent_pairs']])
        h1, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1.to(device))
        h2, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2.to(device))

        predictions = utils.predict_sts_score(
                utils.average_hiddens(h1, sl1.to(device)), 
                utils.average_hiddens(h2, sl2.to(device)),
                h_size=args.h_size,
                syn_size=args.syn_size)
        
        correlation = utils.sts_scoring(predictions, dev_ss['targets'])

    return correlation


def gradient_update(parser):
    update = '\n'.join(['{:35} {:3.8f}'.format(n, p.grad.data.norm(2).item()) for n, p in list(parser.BiLSTM.named_parameters())])

    return update
