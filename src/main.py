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

from parser import BiaffineParser
from data_utils import get_dataset, custom_data_loader, word_dropout

NUM_EPOCHS = 100
BATCH_SIZE = 100  #As Jabberwocky paper stated
H_SIZE = 400

WEIGHTS_DIR = '../weights'
LOG_DIR = '../log'
DATA_DIR = '../data'
MODEL_NAME = 'makeanargparser.tch'
CONLLU_FILE = 'treebank.conllu'
PARANMT_FILE = 'para-nmt-processed-5m.txt'

PAD_TOKEN = '<pad>' # XXX Weird to have out here

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'train.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def train():
    # Set up begins
    INIT_DATA = True
    INIT_MODEL = True
    TRAINING = True

    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocabs_pkl = os.path.join(DATA_DIR, 
            f'{os.path.splitext(CONLLU_FILE)[0]}_vocabs.pkl')
    data_sdp_pkl = os.path.join(DATA_DIR,
            f'{os.path.splitext(CONLLU_FILE)[0]}_data.pkl')
    data_ss_pkl = os.path.join(DATA_DIR,
            f'{os.path.splitext(PARANMT_FILE)[0]}_data.pkl')
    if not os.path.exists(vocabs_pkl) \
            or not os.path.exists(data_sdp_pkl) \
            or not os.path.exists(data_ss_pkl):
        INIT_DATA = True

    if INIT_DATA:
        data_sdp, x2i_maps, i2x_maps, word_counts = get_dataset_sdp(
                os.path.join(DATA_DIR, CONLLU_FILE), training=True)
        data_ss = get_dataset_ss(os.path.join(DATA_DIR, PARANMT_FILE))
        with open(vocabs_pkl, 'wb') as f:
            pickle.dump((x2i_maps, i2x_maps), f)
        with open(data_sdp_pkl, 'wb') as f:
            pickle.dump((data_sdp, word_counts), f)
        with open(data_ss_pkl, 'wb') as f:
            pickle.dump((data_ss), f)
    else:
        with open(vocabs_pkl, 'rb') as f:
            x2i_maps, i2x_maps = pickle.load(f)
        with open(data_sdp_pkl, 'rb') as f:
            data_sdp, word_counts = pickle.load(f)
        with open(data_ss_pkl, 'rb') as f:
            data_ss = pickle.load(f)

    train_sdp = data_sdp['train']
    dev = data_sdp['dev']

    w2i = x2i_maps['word']
    p2i = x2i_maps['pos']
    r2i = x2i_maps['rel']

    i2w = i2x_maps['word']

    parser = BiaffineParser(
            word_vocab_size = len(w2i),
            pos_vocab_size = len(p2i),
            num_relations = len(r2i),
            hidden_size = H_SIZE,
            padding_idx = w2i[PAD_TOKEN])
    parser.to(device)

    model_weights = os.path.join(WEIGHTS_DIR, MODEL_NAME)

    if not INIT_MODEL and os.path.exists(model_weights):
        parser.load_state_dict(torch.load(model_weights))

    # Set up finished

    if TRAINING:
        log.info(f'There are {len(train_data)} training examples.')
        log.info(f'There are {len(dev_data)} validation examples.')

        train_sdp_loader = sdp_data_loader(train_sdp, BATCH_SIZE)
        train_ss_loader = ss_data_loader(train_ss, BATCH_SIZE)
        dev_loader = sdp_data_loader(dev, BATCH_SIZE)

        n_train_batches = ceil(len(train_data) / BATCH_SIZE)
        n_dev_batches = ceil(len(dev_data) / BATCH_SIZE)

        opt = Adam(parser.parameters(), lr=2e-3, betas=[0.9, 0.9])

        earlystop_counter = 0
        prev_best = 0
        log.info('Starting train loop.')

        # For weight analysis
        state = parser.state_dict()

        for e in range(NUM_EPOCHS):

            parser.train()
            train_loss = 0
            for b in range(n_train_batches):

                # Checking to see weights are changing
                log.info('Attention h_rel_head:', state['BiAffineAttention.h_rel_head.0.weight'])
                log.info('Word embedding weight:', state['BiLSTM.word_emb.weight'])

                # Parser training step
                opt.zero_grad()

                words, pos, sent_lens, head_targets, rel_targets = next(train_sdp_loader)
                words_d = word_dropout(words, w2i=w2i, i2w=i2w, counts=word_counts, lens=sent_lens)

                outputs, _ = parser.BiLSTM(words.to(device), pos.to(device), sent_lens)
                outputs_d, _ = parser.BiLSTM(words_d.to(device), pos.to(device), sent_lens)

                # Splice
                outputs[:,:,H_SIZE/2:H_SIZE] = outputs_d[:,:,H_SIZE/2:H_SIZE]
                outputs[:,:,H_SIZE+(H_SIZE/2):] = outputs_d[:,:,H_SIZE+(H_SIZE/2):]

                S_arc, S_rel, _ = parser.BiAffineAttention(outputs.to(device), sent_lens)

                loss_h = loss_heads(S_arc, head_targets)
                loss_r = loss_rels(S_rel, rel_targets)
                loss = loss_h + loss_r

                train_loss += loss_h.item() + loss_r.item()

                loss.backward()
                opt.step()

                # Sentence similarity training step
                opt.zero_grad()

                words, pos, sent_lens = next(train_ss_loader)

                H1, _ = parser.BiLSTM(words[0], pos[0], sent_lens[0])
                H2, _ = parser.BiLSTM(words[1], pos[1], sent_lens[1])

                H1, H2 = average_hiddens(H1,H2)

                loss = loss_ss(H1, H2, T)

                loss.backward()
                opt.step()

            train_loss /= n_train_batches

            parser.eval()  # Crucial! Toggles dropout effects
            dev_loss = 0
            UAS = 0
            LAS = 0
            for b in range(n_dev_batches):
                with torch.no_grad():
                    words, pos, sent_lens, head_targets, rel_targets = next(dev_loader)

                S, L, head_preds = parser(words, pos, sent_lens)
                rel_preds = predict_relations(L, head_preds)

                loss_h = loss_heads(S, head_targets)
                loss_r = loss_rels(L, rel_targets)
                dev_loss = loss_h.item() + loss_r.item()

                UAS, LAS = attachment_scoring(
                        head_preds,
                        rel_preds,
                        head_targets,
                        rel_targets,
                        sent_lens)

            dev_loss /= n_dev_batches

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
        if not os.path.isdir(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
        torch.save(parser.state_dict(), model_weights)

    if TEST:
        #Generate CONLLU file, run test script
        pass

    # End main


def average_hiddens(H1, H2, sent_lens):
    H1 = H1.sum(axis=1)
    H2 = H2.sum(axis=1)

    #sent_lens[0] = torch.Tensor(sent_lens[0]).view(-1, 1)
    sent_lens = torch.Tensor(sent_lens).view(2, -1, 1)  # Column vector

    H1 / sent_lens[0]
    H2 / sent_lens[1]

    return H1, H2

#Eventually, probably should pull loss functions out into a utilities file
def loss_heads(S, head_targets, pad_idx=-1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)
    '''
    # For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S.permute(0,2,1), head_targets, ignore_index=pad_idx)


def loss_rels(L, rel_targets, pad_idx=-1):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''

    return F.cross_entropy(L.permute(0,2,1), rel_targets, ignore_index=pad_idx)


def loss_ss(s, s_para, s_neg):
    margin = 0.4 # As stated in ParaNMT paper
    para_attract = F.cosine_similarity(s, s_para)
    neg_repel = F.cosine_similarity(s, s_neg)

    return F.relu(margin - para_attract + neg_repel)


def predict_relations(L, head_preds):
    '''
    args
        L::Tensor - label logits with shape (b, l, num_rels)

    returns
        rel_preds - shape (b, l)
    '''

    rel_preds = L.argmax(2).long()
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




if __name__ == '__main__':
    options = parse_args
    train()
