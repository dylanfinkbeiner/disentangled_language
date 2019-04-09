import sys
import os
import time
from time import sleep
import logging
import pickle
#from memory_profiler import profile
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from data_utils import sdp_data_loader, word_dropout, ss_data_loader, prepare_batch_ss
from utils import attachment_scoring, get_triplets, average_hiddens, predict_relations
from losses import loss_heads, loss_rels, loss_sem_rep, loss_syn_rep


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
    train_mode = args.trainingmode
    batch_size = args.batchsize
    mega_size = args.M
    h_size = args.hsize
    syn_size = args.synsize
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
        train_ss = train_ss[:len(train_sdp)] #XXX THIS IS STUPID! (but good for testing)
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

    opt = Adam(parser.parameters(), lr=2e-3, betas=[0.9, 0.9])

    earlystop_counter = 0
    prev_best = 0
    log.info('Starting train loop.')
    exp_file.write('Training results:')
    state = parser.state_dict() # For weight analysis
    grad_print = 10
    try:
        for e in range(n_epochs):
            log.info(f'Entering epoch {e+1}/{n_epochs}.')

            parser.train()
            train_loss = 0
            num_steps = 0
            if train_mode == 0:
                for b in range(n_train_batches):
                    log.info(f'Training batch {b+1}/{n_train_batches}.')
                    opt.zero_grad()
                    batch = next(train_sdp_loader)
                    head_targets = batch['head_targets']
                    rel_targets = batch['rel_targets']
                    sent_lens = batch['sent_lens'].to(device)
                    words_d = word_dropout(batch['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=sent_lens)
                    
                    S_arc, S_rel, _ = parser(words_d.to(device), batch['pos'].to(device), sent_lens)

                    loss_h = loss_heads(S_arc, head_targets)
                    loss_r = loss_rels(S_rel, rel_targets)
                    loss = loss_h + loss_r

                    train_loss += loss_h.item() + loss_r.item()

                    loss.backward()
                    opt.step()
                    num_steps += 1

            elif train_mode >= 1:
                for m in range(n_megabatches):
                    megabatch = []
                    idxs = []
                    idx = 0
                    for _ in range(mega_size):
                        instances = [train_ss[j] for j in next(train_ss_loader)]
                        curr_idxs = [i + idx for i in range(len(instances))]
                        megabatch.extend(instances)
                        idxs.append(curr_idxs)
                        idx += len(curr_idxs)

                    with torch.no_grad():
                        s1, s2, negs = get_triplets(megabatch, batch_size, parser, device)

                    for x in range(len(idxs)):
                        print(f'Epoch {e+1}/{n_epochs}, megabatch {m+1}/{n_megabatches}, batch {x+1}/{len(idxs)}')

                        opt.zero_grad()
                        if train_mode == 1:
                            batch, paired, scores = next(train_sdp_loader)
                            batch_lens = batch['sent_lens'].to(device)
                            paired_lens = paired['sent_lens'].to(device)

                            words_d_batch = word_dropout(batch['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=batch_lens)
                            words_d_paired = word_dropout(paired['words'], w2i=w2i, i2w=i2w, counts=word_counts, lens=paired_lens)

                            outputs_batch, _ = parser.BiLSTM(words_d_batch.to(device), batch['pos'].to(device), batch_lens)
                            outputs_paired, _ = parser.BiLSTM(words_d_paired.to(device), paired['pos'].to(device), paired_lens)

                            S_arc_batch, S_rel_batch, _ = parser.BiAffineAttention(outputs_batch.to(device), batch_lens)
                            S_arc_paired, S_rel_paired, _ = parser.BiAffineAttention(outputs_paired.to(device), paired_lens)

                            loss_h_batch = loss_heads(S_arc_batch, batch['head_targets'])
                            loss_r_batch = loss_rels(S_rel_batch, batch['rel_targets'])
                            loss_batch = loss_h_batch + loss_r_batch
                            loss_h_paired = loss_heads(S_arc_paired, paired['head_targets'])
                            loss_r_paired = loss_rels(S_rel_paired, paired['rel_targets'])
                            loss_paired = loss_h_paired + loss_r_paired

                            loss_rep = loss_syn_rep(
                                    average_hiddens(outputs_batch, batch_lens),
                                    average_hiddens(outputs_paired, paired_lens),
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
                            [str(p.grad.data.norm(2).item()) for p in list(parser.BiLSTM.parameters())])
                            log.info(gradient_update)
                            exp_file.write(gradient_update)

                        opt.step()

                        # Sentence similarity step begins
                        opt.zero_grad()

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
                            [str(p.grad.data.norm(2).item()) for p in list(parser.BiLSTM.parameters())])
                            log.info(gradient_update)
                            exp_file.write(gradient_update)

                        opt.step()

            train_loss /= (num_steps if num_steps > 0 else -1)# Just dependency parsing loss

            
            parser.eval()  # Crucial! Toggles dropout effects
            dev_loss = 0
            UAS = 0
            LAS = 0
            total = 0
            log.info('Evaluation step begins.')
            for b in range(n_dev_batches):
                #log.info(f'Eval batch {b+1}/{n_dev_batches}.')
                with torch.no_grad():
                    batch = next(dev_loader)
                    head_targets = batch['head_targets']
                    rel_targets = batch['rel_targets']
                    sent_lens = batch['sent_lens'].to(device)

                    S_arc, S_rel, head_preds = parser(
                            batch['words'].to(device), 
                            batch['pos'].to(device), 
                            sent_lens)
                    rel_preds = predict_relations(S_rel)

                    loss_h = loss_heads(S_arc, head_targets)
                    loss_r = loss_rels(S_rel, rel_targets)
                    dev_loss += loss_h.item() + loss_r.item()

                    #UAS_, LAS_ = attachment_scoring(
                    results = attachment_scoring(
                            head_preds=head_preds.cpu(),
                            rel_preds=rel_preds,
                            head_targets=head_targets,
                            rel_targets=rel_targets,
                            sent_lens=sent_lens,
                            include_root=True)
                    UAS += results['UAS_correct']
                    LAS += results['LAS_correct']
                    total += results['total_words']

            dev_loss /= n_dev_batches
            UAS /= total
            LAS /= total

            update = '''Epoch: {:}
                    Train Loss: {:.3f}
                    Dev Loss: {:.3f}
                    UAS: {:.3f}
                    LAS: {:.3f} '''.format(e, train_loss, dev_loss, UAS, LAS)
            log.info(update)
            exp_file.write(update)

            # Early stopping heuristic from Jabberwocky paper
            if LAS > prev_best:
                earlystop_counter = 0
                prev_best = LAS
            else:
                earlystop_counter += 1
                if earlystop_counter >= 5:
                    print('''LAS has not improved for 5 consecutive epochs,
                          stopping after {} epochs'''.format(e))
                    break
            
            if train_mode != -1:
                torch.save(parser.state_dict(), weights_path)
                log.info(f'Weights saved to {weights_path}.')

    # Save weights
    except KeyboardInterrupt:
        response = input("Keyboard interruption: Would you like to save weights? [y/n]")
        if response.lower() == 'y':
            torch.save(parser.state_dict(), weights_path)
        exp_file.write(f'\n\nExperiment halted by keyboard interrupt. Weights saved : {response.lower()}')
        exp_file.close()

    exp_file.close()
