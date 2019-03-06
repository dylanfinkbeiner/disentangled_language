import sys
import os
import time
import logging
import pickle
from memory_profiler import profile
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from parser import BiaffineParser
from data_utils import get_dataset_sdp, sdp_data_loader, word_dropout
from data_utils import get_dataset_ss, ss_data_loader, prepare_batch_ss
from args import get_args


LOG_DIR = '../log'
DATA_DIR = '../data'
MODEL_NAME = ''
CONLLU_FILE = 'treebank.conllu'
#CONLLU_FILE = 'tenpercentsample.conllu'
PARANMT_FILE = 'para_tiny.txt'

PAD_TOKEN = '<pad>' # XXX Weird to have out here

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'main.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def train(args, parser, data, weights_path=None):
    seed = args.seed
    model_name = args.model
    train_mode = args.train_mode
    init_model = args.initmodel
    batch_size = args.batchsize
    mega_size = args.M
    h_size = args.numhidden
    n_epochs = args.epochs

    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x2i = data['vocabs']['x2i']
    i2x = data['vocabs']['i2x']
    data_sdp = data['data_sdp']

    train_ss = data['data_ss']
    train_sdp = data_sdp['train']
    dev = data_sdp['dev']

    w2i = x2i['word']
    p2i = x2i['pos']
    r2i = x2i['rel']

    i2w = i2x['word']

    # Set up finished

    log.info(f'There are {len(train_sdp)} SDP training examples.')
    log.info(f'There are {len(train_ss)} SS training examples.')
    log.info(f'There are {len(dev)} validation examples.')

    train_sdp_loader = sdp_data_loader(train_sdp, batch_size=batch_size, shuffle=True)
    train_ss_loader = ss_data_loader(train_ss, batch_size=batch_size)
    dev_loader = sdp_data_loader(dev, batch_size=batch_size)

    n_train_batches = ceil(len(train_sdp) / batch_size)
    n_megabatches = ceil(len(train_sdp) / (mega_size * batch_size))
    n_dev_batches = ceil(len(dev) / batch_size)

    opt = Adam(parser.parameters(), lr=2e-3, betas=[0.9, 0.9])

    earlystop_counter = 0
    prev_best = 0
    log.info('Starting train loop.')

    state = parser.state_dict() # For weight analysis

    for e in range(n_epochs):

        parser.train()
        train_loss = 0
        num_steps = 0
        if train_mode == 0:
            for b in range(n_train_batches):
                opt.zero_grad()
                words, pos, sent_lens, head_targets, rel_targets = next(train_sdp_loader)
                words_d = word_dropout(words, w2i=w2i, i2w=i2w, counts=word_counts, lens=sent_lens)
                
                S_arc, S_rel, _ = parser(words_d, pos, sent_lens)

                loss_h = loss_heads(S_arc, head_targets)
                loss_r = loss_rels(S_rel, rel_targets)
                loss = loss_h + loss_r

                train_loss += loss_h.item() + loss_r.item()

                loss.backward()
                opt.step()
                num_steps += 1

        else if train_mode == 1:
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
                    s1, s2, negs = get_triplets(megabatch, batch_size, parser)

                # Checking to see weights are changing
                #log.info('Attention h_rel_head:', state['BiAffineAttention.h_rel_head.0.weight'])
                #log.info('Word embedding weight:', state['BiLSTM.word_emb.weight'])

                for x in range(len(idxs)):
                    opt.zero_grad()

                    words, pos, sent_lens, head_targets, rel_targets = next(train_sdp_loader)
                    words_d = word_dropout(words, w2i=w2i, i2w=i2w, counts=word_counts, lens=sent_lens)

                    outputs, _ = parser.BiLSTM(words.to(device), pos.to(device), sent_lens)
                    outputs_d, _ = parser.BiLSTM(words_d.to(device), pos.to(device), sent_lens)

                    outputs[:,:,h_size // 2 : h_size] = outputs_d[:,:,h_size // 2 : h_size] # Splice forward hiddens
                    outputs[:,:,h_size + (h_size // 2):] = outputs_d[:,:,h_size + (h_size // 2):] # Splice backward hiddens

                    S_arc, S_rel, _ = parser.BiAffineAttention(outputs.to(device), sent_lens)

                    loss_h = loss_heads(S_arc, head_targets)
                    loss_r = loss_rels(S_rel, rel_targets)
                    loss = loss_h + loss_r

                    train_loss += loss_h.item() + loss_r.item()

                    loss.backward()
                    opt.step()
                    num_steps += 1

                    log.info('Sentence similarity training step begins.')
                    opt.zero_grad()

                    w1, p1, sl1 = prepare_batch_ss([s1[i] for i in idxs[x]])
                    w2, p2, sl2 = prepare_batch_ss([s2[i] for i in idxs[x]])
                    wn, pn, sln = prepare_batch_ss([negs[i] for i in idxs[x]])

                    h1, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1)
                    h2, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2)
                    hn, _ = parser.BiLSTM(wn.to(device), pn.to(device), sln)

                    loss = loss_ss(
                            average_hiddens(h1, sl1), 
                            average_hiddens(h2, sl2),
                            average_hiddens(hn, sln))

                    loss.backward()
                    opt.step()

        train_loss /= num_steps # Just dependency parsing loss

        parser.eval()  # Crucial! Toggles dropout effects
        dev_loss = 0
        UAS = 0
        LAS = 0
        for b in range(n_dev_batches):
            with torch.no_grad():
                words, pos, sent_lens, head_targets, rel_targets = next(dev_loader)
                S_arc, S_rel, head_preds = parser(words.to(device), pos.to(device), sent_lens)
                rel_preds = predict_relations(S_rel, head_preds)

                loss_h = loss_heads(S_arc, head_targets)
                loss_r = loss_rels(S_rel, rel_targets)
                dev_loss += loss_h.item() + loss_r.item()

                UAS_, LAS_ = attachment_scoring(
                        head_preds,
                        rel_preds,
                        head_targets,
                        rel_targets,
                        sent_lens)
                UAS += UAS_
                LAS += LAS_

        dev_loss /= n_dev_batches
        UAS /= n_dev_batches
        LAS /= n_dev_batches

        update = '''Epoch: {:}\t
                Train Loss: {:.3f}\t
                Dev Loss: {:.3f}\t
                UAS: {:.3f}\t
                LAS: {:.3f} '''.format(e, train_loss, dev_loss, UAS, LAS)
        log.info(update)

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

    # Save weights
    torch.save(parser.state_dict(), weights_path)


def average_hiddens(hiddens, sent_lens):
    '''
        inputs:
            hiddens - tensor w/ shape (b, l, 2*d) where d is LSTM hidden size
            sent_lens - list of sentence lengths

        outputs:
            averaged_hiddens - 
    '''

    #NOTE WE ARE ASSUMING PAD VALUES ARE 0 IN THIS SUM (NEED TO DOUBLE CHECK)
    averaged_hiddens = hiddens.sum(dim=1)

    sent_lens = torch.Tensor(sent_lens).view(-1, 1).float()  # Column vector

    averaged_hiddens /= sent_lens

    return averaged_hiddens


def get_triplets(megabatch, minibatch_size, parser):
    '''
        inputs:
            megabatch - an unprepared megabatch (M many batches) of sentences
            batch_size - size of a minibatch

        outputs:
            s1 - list of orig. sentence instances
            s2 - list of paraphrase instances
            negs - list of neg sample instances
    '''
    s1 = []
    s2 = []
    
    for mini in megabatch:
        s1.append(mini[0]) # Does this allocate new memory?
        s2.append(mini[1])

    minibatches = [s1[i:i + minibatch_size] for i in range(0, len(s1), minibatch_size)]

    megabatch_of_reps = [] # (megabatch_size, )
    for m in minibatches:
        words, pos, sent_lens = prepare_batch_ss(m)

        m_reps, _ = parser.BiLSTM(words, pos, sent_lens)
        megabatch_of_reps.append(average_hiddens(m_reps, sent_lens))

    megabatch_of_reps = torch.cat(megabatch_of_reps)

    negs = get_negative_samps(megabatch, megabatch_of_reps)

    return s1, s2, negs


def get_negative_samps(megabatch, megabatch_of_reps):
    '''
        inputs:
            megabatch - a megabatch (list) of sentences
            megabatch_of_reps - a tensor of sentence representations

        outputs:
            neg_samps - a list matching length of input megabatch consisting
                        of sentences
    '''
    negs = []

    reps = []
    sents = []
    for i in range(len(megabatch)):
        (s1, _) = megabatch[i]
        reps.append(megabatch_of_reps[i].cpu().numpy())
        sents.append(s1)

    arr = pdist(reps, 'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i] = 0

    arr = np.argmax(arr, axis=1)

    for i in range(len(megabatch)):
        t = None
        t = sents[arr[i]]

        negs.append(t)

    return negs


def loss_heads(S_arc, head_targets, pad_idx=-1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)
    '''
    # For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S_arc.permute(0,2,1), head_targets, ignore_index=pad_idx)


def loss_rels(S_rel, rel_targets, pad_idx=-1):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''

    return F.cross_entropy(S_rel.permute(0,2,1), rel_targets, ignore_index=pad_idx)


def loss_ss(h1, h2, hn, margin=0.4):
    para_attract = F.cosine_similarity(h1, h2) # (b,2*d), (b,2*d) -> (b)
    neg_repel = F.cosine_similarity(h1, hn)

    losses = F.relu(margin - para_attract + neg_repel) # (b)

    return losses.sum()


def predict_relations(S_rel, head_preds):
    '''
    args
        L::Tensor - label logits with shape (b, l, num_rels)

    returns
        rel_preds - shape (b, l)
    '''

    rel_preds = S_rel.argmax(2).long()
    return rel_preds


def attachment_scoring(head_preds, rel_preds, head_targets, rel_targets, sent_lens):
    '''
        input:
            head_preds::Tensor - Has shape (b, l), -1 padded
            rel_preds::Tensor -  Has shape (b, l, num_rel), -1 padded
            head_targets::Tensor - (-1)-padded (b, l) tensor of ints
            rel_targets::Tensor - (-1)-padded (b, l) tensor of ints


        returns:
            UAS - average number of correct head predictions
            LAS - average number of correct relation predictions
    '''
    sent_lens = torch.Tensor(sent_lens).view(-1, 1)
    b, l = head_preds.shape

    # This way we can be sure padding values do not contribute to score when we do .eq() calls
    head_preds = torch.where(
            head_targets != -1,
            head_preds,
            torch.zeros(head_preds.shape).long())
    rel_preds = torch.where(
            rel_targets != -1,
            rel_preds,
            torch.zeros(rel_preds.shape).long())

    # Tensors with 1s in locations of correct predictions
    #NOTE this could be optimized later to avoid sparse matrices
    correct_heads = head_preds.eq(head_targets).float()
    correct_rels = rel_preds.eq(rel_targets).float()

    UAS = correct_heads.sum(1, True)
    UAS /= sent_lens
    UAS = UAS.sum() / b

    LAS = (correct_heads * correct_rels).sum(1, True)
    LAS /= sent_lens
    LAS = LAS.sum() / b

    return UAS, LAS