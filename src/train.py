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
    # Append to existing training experiment file for given day
    prelude = write_prelude(args, experiment)
    log = get_logger(experiment)
    log.info(prelude)
    sleep(5)

    if 0 in args.train_mode:
        train_sdp = data['data_ptb']['train']
        dev_sdp = data['data_ptb']['dev']
        loader_sdp_train = data_utils.sdp_data_loader(train_sdp, batch_size=args.sdp_bs, shuffle_idx=True)
        loader_sdp_dev = data_utils.sdp_data_loader(dev_sdp, batch_size=100, shuffle_idx=False)
        n_train_batches = ceil(len(train_sdp) / args.sdp_bs)
        n_dev_batches = ceil(len(dev_sdp) / 100)
        log.info(f'There are {len(train_sdp)} syntactic dependency parsing training examples.')
        log.info(f'There are {len(dev_sdp)} syntactic dependency parsing dev examples.')
        sdp_devloss_record = []
        sdp_uas_record = []
        sdp_las_record = []
    if 1 in args.train_mode:
        train_ss = data['data_ss']['train']['sent_pairs']
        dev_ss = data['data_ss']['dev']
        idxloader_ss_train = data_utils.idx_loader(num_data=len(train_ss), batch_size=args.sem_bs)
        n_megabatches = ceil(len(train_ss) / (args.M * args.sem_bs))
        log.info(f'There are {len(train_ss)} semantic similarity training examples.')
        log.info(f"There are {len(dev_ss['sent_pairs'])} semantic similarity dev examples.")
        sem_corr_record = []
    if 2 in args.train_mode:
        train_stag = data['data_stag']['train']
        dev_stag = data['data_stag']['dev']
        loader_stag_train = data_utils.stag_data_loader(train_stag, batch_size=args.stag_bs, shuffle_idx=True)
        loader_stag_dev = data_utils.stag_data_loader(dev_stag, batch_size=100, shuffle_idx=False)
        log.info(f'There are {len(train_stag)} supertagging training examples.')
        log.info(f'There are {len(dev_stag)} supertagging dev examples.')
        stag_acc_record = []
    if args.adv_stag:
        train_adv_stag = data['data_stag']['train']
        loader_adv_stag_train = data_utils.stag_data_loader(train_adv_stag, batch_size=args.stag_bs, shuffle_idx=True)

    opt = Adam(parser.parameters(), lr=args.lr, betas=[0.9, 0.9])

    print_grad_every = 10
    earlystop_counter = 0
    prev_best = 0
    if 0 in args.train_mode:
        sdp_batches = n_train_batches # Counts down to 0, whereafter we check early stopping criterion
        _, prev_best, _ = sdp_dev_eval(parser, args=args, data=data, loader=loader_sdp_dev, n_dev_batches=n_dev_batches)
    try:
        for e in range(args.epochs):
            log.info(f'Entering epoch {e+1}/{args.epochs}.')

            if args.train_mode == [0]:
                for b in tqdm(range(n_train_batches), ascii=True, desc=f'Epoch {e+1}/{args.epochs} progress', ncols=80):
                    opt.zero_grad()
                    batch = next(loader_sdp_train)
                    loss = forward_syntactic_parsing(parser, batch, args=args, data=data)
                    loss.backward()
                    opt.step()

            elif args.train_mode == [2]:
                n_stag_bs = ceil(len(train_stag) / args.stag_bs)
                for b in tqdm(range(n_stag_bs), ascii=True, desc=f'Epoch {e+1}/{args.epochs} progress', ncols=80):
                    opt.zero_grad()
                    stag_batch = next(loader_stag_train)
                    loss_stag = forward_stag(
                            parser,
                            batch=stag_batch,
                            args=args,
                            data=data)
                    loss_stag.backward()
                    opt.step()

            #elif 3 in args.train_mode:
            #    opt.zero_grad()
            #    for p in parser.parameters():
            #        p.requires_grad = True
            #    for p in parser.AdvStagMLP.parameters():
            #        p.requires_grad = False
            #    sdp_batch = next(loader_sdp_train)
            #    stag_batch = next(loader_stag_train)
            #    loss_stag = forward_adversarial_stag(parser, batch=stag_batch, args=args, data=data)
            #    loss_syn = forward_syntactic_parsing(parser, batch=sdp_batch, args=args, data=data)
            #    loss = loss_syn - (args.scale_adv_stag * loss_stag)
            #    loss.backward()
            #    print(gradient_update(parser, verbose=True))
            #    breakpoint()
            #    opt.step()

            #    opt.zero_grad()
            #    for p in parser.parameters():
            #        p.requires_grad = False
            #    for p in parser.AdvStagMLP.parameters():
            #        p.requires_grad = True
            #    stag_batch = next(loader_stag_train)
            #    loss_stag = forward_adversarial_stag(parser, batch=stag_batch, args=args, data=data)
            #    loss_stag.backward()
            #    breakpoint()
            #    opt.step()
            

            elif 1 in args.train_mode:
                if args.adv_stag:
                    for p in parser.parameters():
                        p.requires_grad = True
                    for p in parser.AdvStagMLP.parameters():
                        p.requires_grad = False

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
                        loss = torch.zeros(1).to(data['device'])
                        opt.zero_grad()

                        if 0 in args.train_mode:
                            sdp_batch = next(loader_sdp_train)
                            sdp_batches -= 1
                            loss_sdp = forward_syntactic_parsing(
                                    parser, 
                                    sdp_batch,
                                    args=args, 
                                    data=data)
                            loss += loss_sdp

                        if 2 in args.train_mode:
                            stag_batch = next(loader_stag_train)
                            loss_stag = forward_stag(
                                    parser,
                                    batch=stag_batch,
                                    args=args,
                                    data=data)
                            loss += loss_stag

                        if (e+1) >= args.start_epoch and args.adv_stag:
                            stag_adv_batch = next(loader_adv_stag_train)
                            loss_adv_stag = forward_adversarial_stag(
                                    parser,
                                    batch=stag_batch,
                                    args=args,
                                    data=data)
                            if x % print_grad_every == 0:
                                log.info(f'Adversarial loss: {loss_adv_stag.item()}')
                            loss -= (args.scale_adv_stag * loss_adv_stag)

                        loss_sem = forward_semantic(
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

                        if 0 in args.train_mode and sdp_batches == 0:
                           _, LAS, _ = sdp_dev_eval(
                                   parser, 
                                   args=args, 
                                   data=data, 
                                   loader=loader_sdp_dev,
                                   n_dev_batches=n_dev_batches)
                           prev_best, earlystop_counter, msg = earlystop_check(args, LAS, prev_best, earlystop_counter)
                           log.info(msg)
                           if earlystop_counter >= args.earlystop_pt:
                               break
                           sdp_batches = n_train_batches

                        #if x % print_grad_every == 0:
                        #    print(gradient_update(parser, verbose=False))
                        #print(gradient_update(parser, verbose=False))

                    # End of individual megabatch's loop
                    if earlystop_counter >= args.earlystop_pt:
                        break

                    #update = ''
                    #update += f'''\nUpdate for megabatch: {m+1}\n'''
                    #correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                    #update += '''Correlation: {:.4f}'''.format(correlation)

                    #log.info(update)

                # End of megabatches loop    

            if args.adv_stag:
                stag_accuracy = stag_dev_eval(
                        parser,
                        args=args, 
                        data=data, 
                        loader=loader_stag_dev, 
                        n_dev_batches=ceil(len(dev_stag) / 100),
                        adversarial=True)
                log.info(f'\nAdversarial supertagger accuracy before, epoch {e+1}: {stag_accuracy:.4f}\n')

            if args.adv_stag and not (earlystop_counter >= args.earlystop_pt):
                for p in parser.parameters():
                    p.requires_grad = False
                for p in parser.AdvStagMLP.parameters():
                    p.requires_grad = True
                n_stag_bs = ceil(len(train_adv_stag) / args.stag_bs)
                for b in tqdm(range(n_stag_bs), ascii=True, desc=f'Adversarial stag training for epoch {e+1}/{args.epochs}', ncols=80):
                    opt.zero_grad()
                    adv_stag_batch = next(loader_adv_stag_train)
                    loss_stag = forward_adversarial_stag(
                            parser,
                            batch=adv_stag_batch,
                            args=args,
                            data=data)
                    loss_stag.backward()
                    opt.step()


            update = f'\nUpdate for epoch: {e+1}/{args.epochs}\n'
            if 0 in args.train_mode:
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
            if 1 in args.train_mode:
                correlation = ss_dev_eval(parser, dev_ss, args=args, data=data)
                sem_corr_record.append(correlation)
                update += f'\nCorrelation: {correlation:.4f}\n'
            if 2 in args.train_mode:
                stag_accuracy = stag_dev_eval(
                        parser,
                        args=args, 
                        data=data, 
                        loader=loader_stag_dev, 
                        n_dev_batches=ceil(len(dev_stag) / 100),
                        adversarial=False)
                stag_acc_record.append(stag_accuracy)
                update += f'\nSupertagging accuracy: {stag_accuracy:.4f}\n'
            if args.adv_stag:
                stag_accuracy = stag_dev_eval(
                        parser,
                        args=args, 
                        data=data, 
                        loader=loader_stag_dev, 
                        n_dev_batches=ceil(len(dev_stag) / 100),
                        adversarial=True)
                update += f'\nAdversarial supertagger accuracy after, epoch {e+1}: {stag_accuracy:.4f}\n'


            log.info(update)

            if args.train_mode != [-1]:
                torch.save(parser.state_dict(), weights_path)
                log.info(f'Weights saved to {weights_path}.')

            #if 0 in args.train_mode and not 1 in args.train_mode:
            if args.train_mode == [0]:
                prev_best, earlystop_counter, msg = earlystop_check(args, LAS, prev_best, earlystop_counter)
                log.info(msg)

            if earlystop_counter >= args.earlystop_pt:
                break

            #if args.train_mode == [2]:
            #    if stag_accuracy > prev_best + 0.00005:
            #        msg = f'LAS improved from {prev_best} on epoch {e+1}/{args.epochs}.'
            #        log.info(msg)
            #        earlystop_counter = 0
            #        prev_best = stag_accuracy
            #    else:
            #        earlystop_counter += 1
            #        msg = f'LAS has not improved for {earlystop_counter} consecutive epoch(s).'
            #        log.info(msg)
            #        if earlystop_counter >= args.earlystop_pt:
            #            break




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
                if 0 in args.train_mode:
                    exp_data['sdp_devloss_record'] = sdp_devloss_record
                    exp_data['sdp_uas_record'] = sdp_uas_record
                    exp_data['sdp_las_record'] = sdp_las_record
                if 1 in args.train_mode:
                    exp_data['sem_corr_record'] = sem_corr_record
                if 2 in args.train_mode:
                    exp_data['stag_acc_record'] = stag_acc_record
                pickle.dump(exp_data, pkl)

        d = dt.datetime.now().astimezone(pytz.timezone("America/Los_Angeles"))
        log.info('\n'*3 + f'Experiment ended at {d:%H:%M:%S}\n')


def forward_syntactic_parsing(parser, batch, args=None, data=None):
    parser.train()
    device = data['device']
    sent_lens = batch['sent_lens']
    pos = batch['pos'].to(device) if parser.pos_in else None
    
    words_d, mask = utils.word_dropout(
            batch['words'], 
            w2i=data['vocabs']['x2i']['word'], 
            i2w=data['vocabs']['i2x']['word'], 
            counts=data['word_counts'], 
            lens=sent_lens,
            rate=args.word_dropout,
            style=args.drop_style)

    pos_d, mask = utils.pos_dropout(pos.to('cpu'), lens=sent_lens, p2i=data['vocabs']['x2i']['pos'], p=args.pos_dropout)

    if not args.semantic_dropout:
        S_arc, S_rel, _ = parser(words_d.to(device), sent_lens, pos=pos_d.to(device))
    else:
        S_arc, S_rel, _ = parser(
                batch['words'].to(device), 
                sent_lens, 
                pos=pos_d.to(device), 
                mask=mask)
    
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

    pos_d, mask = utils.pos_dropout(pos.to('cpu'), lens=sent_lens, p2i=data['vocabs']['x2i']['pos'], p=args.pos_dropout)

    if args.wd_stag:
        words_d, mask = utils.word_dropout(
                batch['words'], 
                w2i=data['vocabs']['x2i']['word'], 
                i2w=data['vocabs']['i2x']['word'], 
                counts=data['word_counts'], 
                lens=sent_lens,
                rate=args.word_dropout,
                style=args.drop_style)

        lstm_input, indices, _, _ = parser.Embeddings(
                #batch['words'].to(device), 
                words_d.to(device), 
                sent_lens,
                pos=pos_d.to(device))
    else:
        lstm_input, indices, _, _ = parser.Embeddings(
                batch['words'].to(device), 
                sent_lens,
                pos=pos_d.to(device))

    outputs = parser.SyntacticRNN(lstm_input)
    logits = parser.StagMLP(unsort(outputs, indices))

    loss = losses.loss_stag(logits, stag_targets)
    
    loss *= args.lr_stag

    return loss


def forward_adversarial_stag(parser, batch, args=None, data=None):
    parser.train()
    device = data['device']

    stag_targets = batch['stag_targets'].to(device)
    sent_lens = batch['sent_lens']
    pos = batch['pos'].to(device) if parser.pos_in else None

    pos_d, mask = utils.pos_dropout(pos.to('cpu'), lens=sent_lens, p2i=data['vocabs']['x2i']['pos'], p=args.pos_dropout)

    if args.wd_stag:
        words_d, mask = utils.word_dropout(
                batch['words'], 
                w2i=data['vocabs']['x2i']['word'], 
                i2w=data['vocabs']['i2x']['word'], 
                counts=data['word_counts'], 
                lens=sent_lens,
                rate=args.word_dropout,
                style=args.drop_style)

        lstm_input, indices, _, _ = parser.Embeddings(
                #batch['words'].to(device), 
                words_d.to(device), 
                sent_lens,
                pos=pos_d.to(device))
    else:
        lstm_input, indices, _, _ = parser.Embeddings(
                batch['words'].to(device), 
                sent_lens,
                pos=pos_d.to(device))

    outputs = parser.SemanticRNN(lstm_input)
    logits = parser.AdvStagMLP(unsort(outputs, indices))

    loss = losses.loss_stag(logits, stag_targets)
    
    return loss



def forward_semantic(parser, s1, s2, n1, n2=None, args=None, data=None):
    parser.train()
    device = data['device']

    pi = parser.pos_in

    w2i = data['vocabs']['x2i']['word']
    i2w = data['vocabs']['i2x']['word']

    w1, p1, sl1 = data_utils.prepare_batch_ss(s1)
    w2, p2, sl2 = data_utils.prepare_batch_ss(s2)
    wn1, pn1, sln1 = data_utils.prepare_batch_ss(n1)
    if args.word_dropout > 0. and args.wd_sem:
        counts = data['word_counts']
        w1, _ = utils.word_dropout(w1, w2i=w2i, i2w=i2w, counts=counts, lens=sl1, rate=args.word_dropout, style=args.drop_style)
        w2, _ = utils.word_dropout(w2, w2i=w2i, i2w=i2w, counts=counts, lens=sl2, rate=args.word_dropout, style=args.drop_style)
        wn1, _ = utils.word_dropout(wn1, w2i=w2i, i2w=i2w, counts=counts, lens=sln1, rate=args.word_dropout, style=args.drop_style)
    
    packed_s1, idx_s1, _, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
    packed_s2, idx_s2, _, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
    packed_n1, idx_n1, _, _ = parser.Embeddings(wn1.to(device), sln1, pos=pn1.to(device) if pi else None)

    h1 = unsort(parser.SemanticRNN(packed_s1), idx_s1)
    h2 = unsort(parser.SemanticRNN(packed_s2), idx_s2)
    hn1 = unsort(parser.SemanticRNN(packed_n1), idx_n1)

    h1_avg = utils.average_hiddens(h1, sl1.to(device), sum_f_b=args.sum_f_b)
    h2_avg = utils.average_hiddens(h2, sl2.to(device), sum_f_b=args.sum_f_b)
    hn1_avg = utils.average_hiddens(hn1, sln1.to(device), sum_f_b=args.sum_f_b)

    hn2_avg = None
    if n2 is not None:
        wn2, pn2, sln2 = data_utils.prepare_batch_ss(n2)
        packed_n2, idx_n2, _, _ = parser.Embeddings(wn2.to(device), sln2.to(device), pos=pn2.to(device) if pi else None)
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
            UAS += results['UAS_correct'].item()
            LAS += results['LAS_correct'].item()
            total_words += results['total_words'].item()

    dev_loss /= n_dev_batches
    UAS /= total_words
    LAS /= total_words

    return UAS, LAS, dev_loss


def stag_dev_eval(parser, args=None, data=None, loader=None, n_dev_batches=None, adversarial=False):
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

            lstm_input, indices, _, _ = parser.Embeddings(
                    batch['words'].to(device), 
                    sent_lens, 
                    pos=pos)
            if adversarial:
                outputs = parser.SemanticRNN(lstm_input)
                logits = parser.AdvStagMLP(unsort(outputs, indices))
            else:
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

        packed_s1, idx_s1, _, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
        packed_s2, idx_s2, _, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
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
        update += '\n'.join(['{:35} {:3.8f}'.format(
            n, p.grad.data.norm(2).item()) for n, p in list(parser.named_parameters()) if type(p.grad) != type(None)])
        update += f'\nTotal norm of gradient is {t}\n'
    else: 
        update += 'Total norm of gradient: {:5.8f}\n'.format(t)

    return update


def earlystop_check(args, LAS, prev_best, earlystop_counter):
    if LAS > prev_best + 0.00005:
        msg = f'LAS improved from {prev_best}.'
        earlystop_counter = 0
        prev_best = LAS
    else:
        earlystop_counter += 1
        msg = f'LAS has not improved for {earlystop_counter} consecutive epoch(s).'
    
    return prev_best, earlystop_counter, msg

def param_names(parser):
    for n, _ in list(parser.named_parameters()):
        print(n)
