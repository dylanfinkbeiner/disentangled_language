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

from data_utils import sdp_data_loader, ss_data_loader, prepare_batch_ss, get_triplets
from losses import loss_arcs, loss_rels, loss_sem_rep, loss_syn_rep
from utils import attachment_scoring, average_hiddens, predict_relations, word_dropout


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


def train(args, parser, data, weights_path=None, exp_path_base=None):
    seed = args.seed
    model_name = args.model
    train_mode = args.train_mode
    batch_size = args.batch_size
    mega_size = args.M
    h_size = args.h_size
    syn_size = args.syn_size
    n_epochs = args.epochs if train_mode != -1 else 1
    custom_task = train_mode > 0

    exp_path = '_'.join([exp_path_base, str(train_mode)])
    exp_file = open(exp_path, 'a')
    exp_file.write(f'Training experiment for model : {model_name}')

    log.info(f'Training model \"{model_name}\" for {n_epochs} epochs in training mode {train_mode}.')
    log.info(f'Weights will be saved to {weights_path}.')
    sleep(5)

    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x2i = data['vocabs']['x2i']
    i2x = data['vocabs']['i2x']
    w2i = x2i['word']
    i2w = i2x['word']
    word_counts = data['word_counts']
    data_ptb = data['data_ptb']

    train_sdp = data_ptb['train']
    if train_mode > 0:
        train_ss = data['data_ss']
        train_ss = train_ss[:len(train_sdp)] #XXX 
    dev = data_ptb['dev']

    log.info(f'There are {len(train_sdp)} SDP training examples.')
    if train_mode > 0:
        log.info(f'There are {len(train_ss)} SS training examples.')
    log.info(f'There are {len(dev)} validation examples.')

    train_sdp_loader = sdp_data_loader(train_sdp, batch_size=batch_size, shuffle_idx=True, custom_task=custom_task)
    if train_mode > 0:
        train_ss_loader = ss_data_loader(train_ss, batch_size=batch_size)
    dev_batch_size = batch_size
    dev_loader = sdp_data_loader(dev, batch_size=dev_batch_size, shuffle_idx=False, custom_task=False)

    n_train_batches = ceil(len(train_sdp) / batch_size)
    n_dev_batches = ceil(len(dev) / dev_batch_size)
    if train_mode > 0:
        n_megabatches = ceil(len(train_ss) / (mega_size * batch_size))

    opt_sdp = Adam(parser.parameters(), lr=args.lr_syn, betas=[0.9, 0.9])
    if train_mode > 0:
        opt_ss = Adam(parser.parameters(), lr=args.lr_sem, betas=[0.9, 0.9])


    earlystop_counter = 0
    prev_best = 0
    #prev_best = dev_eval(parser, args=args, data=data, loader=dev_loader) if not args.initmodel
    log.info('Starting train loop.')
    exp_file.write('Training results:')
    state = parser.state_dict() # For weight analysis
    grad_print = 10

    gradient_update = '\n'.join([ '========================'] + [n for n, p in list(parser.named_parameters())])
    log.info(gradient_update)
    gradient_update = '\n'.join([ '========================'] + [n for n, p in list(parser.BiLSTM.named_parameters())])
    log.info(gradient_update)
    parser.train()
    print(f'named: {len(list(parser.BiLSTM.named_parameters()))}, unnamed: {len(list(parser.BiLSTM.parameters()))}')

    exp_file.write(gradient_update)
    try:
        for e in range(n_epochs):
            log.info(f'Entering epoch {e+1}/{n_epochs}.')

            parser.train()
            train_loss = 0
            num_steps = 0
            if train_mode == 0: # Standard syntactic parsing
                for b in tqdm(range(n_train_batches), ascii=True, desc='Batches complete: ', ncols=80):
                    #log.info(f'Training batch {b+1}/{n_train_batches}.')
                    opt.zero_grad()
                    batch = next(train_sdp_loader)
                    arc_targets = batch['arc_targets']
                    rel_targets = batch['rel_targets']
                    sent_lens = batch['sent_lens'].to(device)
                    words_d = word_dropout(batch['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=sent_lens)
                    
                    S_arc, S_rel, _ = parser(words_d.to(device), batch['pos'].to(device), sent_lens)

                    loss_h = loss_arcs(S_arc, arc_targets)
                    loss_r = loss_rels(S_rel, rel_targets)
                    loss = loss_h + loss_r

                    train_loss += loss_h.item() + loss_r.item()

                    loss.backward()
                    opt_sdp.step()
                    num_steps += 1

            elif train_mode == 1:
                for m in range(n_megabatches):
                    megabatch = []
                    idxs = []
                    idx = 0
                    for _ in range(mega_size):
                        mini_batch = [train_ss[j] for j in next(train_ss_loader)]
                        curr_idxs = [i + idx for i in range(len(mini_batch))]
                        megabatch.extend(mini_batch)
                        idxs.append(curr_idxs)
                        idx += len(curr_idxs)

                    with torch.no_grad():
                        s1, s2, negs = get_triplets(megabatch, batch_size, parser, device)

                    for x in range(len(idxs)):
                        log.info(f'Epoch {e+1}/{n_epochs}, megabatch {m+1}/{n_megabatches}, batch {x+1}/{len(idxs)}')

                        opt_sdp.zero_grad()
                        if train_mode == 1:
                            batch, paired, scores = next(train_sdp_loader)
                            batch_lens = batch['sent_lens'].to(device)
                            paired_lens = paired['sent_lens'].to(device)

                            words_d_batch = word_dropout(batch['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=batch_lens)
                            words_d_paired = word_dropout(paired['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=paired_lens)

                            h_batch, _ = parser.BiLSTM(words_d_batch.to(device), batch['pos'].to(device), batch_lens)
                            h_paired, _ = parser.BiLSTM(words_d_paired.to(device), paired['pos'].to(device), paired_lens)

                            S_arc_batch, S_rel_batch, _ = parser.BiAffineAttention(h_batch.to(device), batch_lens)
                            S_arc_paired, S_rel_paired, _ = parser.BiAffineAttention(h_paired.to(device), paired_lens)

                            loss_h_batch = loss_arcs(S_arc_batch, batch['arc_targets'])
                            loss_r_batch = loss_rels(S_rel_batch, batch['rel_targets'])
                            loss_batch = loss_h_batch + loss_r_batch
                            loss_h_paired = loss_arcs(S_arc_paired, paired['arc_targets'])
                            loss_r_paired = loss_rels(S_rel_paired, paired['rel_targets'])
                            loss_paired = loss_h_paired + loss_r_paired

                            if x % grad_print == 0:
                                print('scores look like: ', scores)

                            loss_rep = loss_syn_rep(
                                    average_hiddens(h_batch, batch_lens),
                                    average_hiddens(h_paired, paired_lens),
                                    scores.to(device), 
                                    syn_size=syn_size,
                                    h_size=h_size)

                            loss = loss_batch.to(device) + loss_paired.to(device) + loss_rep

                            train_loss += loss.item()
                            num_steps += 1
                            

                        loss.backward()
                        if x % grad_print == 0:
                            gradient_update = '\n'.join([ '========================',
                            f'Epoch {e+1}/{n_epochs}, megabatch {m+1}/{n_megabatches}, batch {x+1}/{len(idxs)}',
                            'Gradients after SENTENCE SIMILARITY step:'] +
                            ['{:35} {:3.8f}'.format(n, p.grad.data.norm(2).item()) for n, p in list(parser.BiLSTM.named_parameters())])
                            log.info(gradient_update)
                            exp_file.write(gradient_update)

                        opt_sdp.step()

                        # Sentence similarity step begins
                        opt_ss.zero_grad()

                        w1, p1, sl1 = prepare_batch_ss([s1[i] for i in idxs[x]])
                        w2, p2, sl2 = prepare_batch_ss([s2[i] for i in idxs[x]])
                        wn, pn, sln = prepare_batch_ss([negs[i] for i in idxs[x]])

                        h1, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1.to(device))
                        h2, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2.to(device))
                        hn, _ = parser.BiLSTM(wn.to(device), pn.to(device), sln.to(device))

                        loss = loss_sem_rep(
                                average_hiddens(h1, sl1.to(device)),
                                average_hiddens(h2, sl2.to(device)),
                                average_hiddens(hn, sln.to(device)),
                                syn_size=syn_size,
                                h_size=h_size)

                        loss.backward()
                        if x % grad_print == 0:
                            gradient_update = '\n'.join([ '========================',
                            f'Epoch {e+1}/{n_epochs}, megabatch {m+1}/{n_megabatches}, batch {x+1}/{len(idxs)}',
                            'Gradients after SENTENCE SIMILARITY step:'] +
                            [f'{p.name}:' + str(p.grad.data.norm(2).item()) for p in list(parser.BiLSTM.parameters())])
                            log.info(gradient_update)
                            exp_file.write(gradient_update)

                        opt_ss.step()

            train_loss /= (num_steps if num_steps > 0 else -1) # Just dependency parsing loss

            
            parser.eval()  # Crucial! Toggles dropout effects
            dev_loss = 0
            UAS = 0
            LAS = 0
            total_words = 0
            log.info('Evaluation step begins.')
            for b in range(n_dev_batches):
                #log.info(f'Eval batch {b+1}/{n_dev_batches}.')
                with torch.no_grad():
                    batch = next(dev_loader)
                    arc_targets = batch['arc_targets']
                    rel_targets = batch['rel_targets']
                    sent_lens = batch['sent_lens'].to(device)

                    S_arc, S_rel, arc_preds = parser(
                            batch['words'].to(device), 
                            batch['pos'].to(device), 
                            sent_lens)
                    rel_preds = predict_relations(S_rel)

                    loss_h = loss_arcs(S_arc, arc_targets)
                    loss_r = loss_rels(S_rel, rel_targets)
                    dev_loss += loss_h.item() + loss_r.item()

                    results = attachment_scoring(
                            arc_preds=arc_preds.cpu(),
                            rel_preds=rel_preds,
                            arc_targets=arc_targets,
                            rel_targets=rel_targets,
                            sent_lens=sent_lens,
                            include_root=True)
                    UAS += results['UAS_correct']
                    LAS += results['LAS_correct']
                    total_words += results['total_words']

            dev_loss /= n_dev_batches
            UAS /= total_words
            LAS /= total

            update = '''Epoch: {:}
                    Train Loss: {:.3f}
                    Dev Loss: {:.3f}
                    UAS: {:.3f}
                    LAS: {:.3f}'''.format(e, train_loss, dev_loss, UAS * 100, LAS * 100)
            log.info(update)
            exp_file.write(update)

            # Early stopping heuristic from Jabberwocky paper
            if LAS > prev_best:
                print('LAS improved.')
                earlystop_counter = 0
                prev_best = LAS
            else:
                earlystop_counter += 1
                print(f'LAS has not improved for {earlystop_counter} consecutive epochs.')
                if earlystop_counter >= 5:
                    print(f'Stopping after {e} epochs')
                    break
            
            if train_mode != -1:
                torch.save(parser.state_dict(), weights_path)
                log.info(f'Weights saved to {weights_path}.')

    except KeyboardInterrupt:
        response = input("Keyboard interruption: Would you like to save weights? [y/n]")
        if response.lower() == 'y':
            torch.save(parser.state_dict(), weights_path)
        exp_file.write(f'\n\nExperiment halted by keyboard interrupt. Weights saved : {response.lower()}')
        exp_file.close()

    exp_file.close()


def train_standard_parser(parser, args=None, data=None, opt=None, loader=None):
    parser.train()
    for b in range(n_training_batches):
       opt.zero_grad()
       batch = next(loader)
       arc_targets = batch['arc_targets']
       rel_targets = batch['rel_targets']
       sent_lens = batch['sent_lens'].to(device)

       words_d = word_dropout(
               batch['words'], 
               w2i=data['vocabs']['x2i']['word'], 
               i2w=data['vocabs']['i2x']['word'], 
               counts=data['word_counts'], 
               lens=sent_lens)
       
       S_arc, S_rel, _ = parser(words_d.to(device), batch['pos'].to(device), sent_lens)
    
       loss_h = loss_arcs(S_arc, arc_targets)
       loss_r = loss_rels(S_rel, rel_targets)
       loss = loss_h + loss_r
    
       train_loss += loss_h.item() + loss_r.item()
    
       loss.backward()
       opt.step()
       #num_steps += 1


def dev_eval(parser, args=None, data=None, loader=None):
    parser.eval()  # Crucial! Toggles dropout effects
    dev_loss = 0
    UAS = 0
    LAS = 0
    total_words = 0
    log.info('Evaluation step begins.')
    for b in range(n_dev_batches):
        with torch.no_grad():
            batch = next(loader)
            arc_targets = batch['arc_targets']
            rel_targets = batch['rel_targets']
            sent_lens = batch['sent_lens'].to(device)
    
            S_arc, S_rel, arc_preds = parser(
                    batch['words'].to(device), 
                    batch['pos'].to(device), 
                    sent_lens)
            rel_preds = predict_relations(S_rel)
    
            loss_h = loss_arcs(S_arc, arc_targets)
            loss_r = loss_rels(S_rel, rel_targets)
            dev_loss += loss_h.item() + loss_r.item()
    
            results = attachment_scoring(
                    arc_preds=arc_preds.cpu(),
                    rel_preds=rel_preds,
                    arc_targets=arc_targets,
                    rel_targets=rel_targets,
                    sent_lens=sent_lens,
                    include_root=True)
            UAS += results['UAS_correct']
            LAS += results['LAS_correct']
            total_words += results['total_words']

        dev_loss /= n_dev_batches
        UAS /= total_words
        LAS /= total_words

        update = '''Epoch: {:}
                Train Loss: {:.3f}
                Dev Loss: {:.3f}
                UAS: {:.3f}
                LAS: {:.3f}'''.format(e, train_loss, dev_loss, UAS * 100, LAS * 100)

        log.info(update)
        exp_file.write(update)
