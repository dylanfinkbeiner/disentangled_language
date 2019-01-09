import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

from parser import BiaffineParser
from ptb_dataset import get_dataset, custom_data_loader

from math import ceil

import sys

NUM_EPOCHS = 100
BATCH_SIZE = 10 #As Jabberwocky paper stated

WEIGHTS_PATH = '../weights'
MODEL_NAME = 'newmodel'

CONLLU_FILE = './data/treebank.conllu'

train = True

PAD_TOKEN = '<pad>'

'''
    What remains to be done:
    1. Train/test splitting
    2. During-training evaluation
    3. Code for getting results for test data
    4. Confirm cross entropy loss being calculated properly
    5. Assessing padding situation (i.e. shall we account for it in loss equation?)

'''

def main():
    #if os.path.isdir(WEIGHTS_PATH):
    #    parser.load_state_dict(torch.load(WEIGHTS_PATH))

    if train:
        data_split, x2num_maps, num2x_maps, word_counts = get_dataset(CONLLU_FILE, training = True)

        train_data = data_split['train']
        dev_data = data_split['dev']

        word2num = x2num_maps['word']
        pos2num = x2num_maps['pos']
        rel2num = x2num_maps['rel']

        num2word = num2x_maps['word']

        train_loader = custom_data_loader(train_data, BATCH_SIZE, word2num, num2word, word_counts) 
        dev_loader = custom_data_loader(dev_data, BATCH_SIZE, word2num, num2word, word_counts) 

        parser = BiaffineParser(
                word_vocab_size = len(word2num),
                pos_vocab_size = len(pos2num),
                num_relations = len(rel2num),
                padding_idx = word2num[PAD_TOKEN])

        n_train_batches = ceil(len(train_data) / BATCH_SIZE)
        n_dev_batches = ceil(len(dev_data) / BATCH_SIZE)

        # Optimizer (still not doing the exponential annealing decay thing from Dozat/Manning
        opt = Adam(parser.parameters(), lr = 2e-3, betas=[0.9, 0.9])

        earlystop_counter = 0
        prev_best = 0
        for e in range(NUM_EPOCHS):

            parser.train()
            train_loss = 0
            for b in range(n_train_batches):
                opt.zero_grad()

                words, pos, sent_lens, target_heads, target_rels = next(train_loader)

                S, L, _ = parser(words, pos, sent_lens)

                loss_h = loss_heads(S, target_heads)
                loss_r = loss_rels(L, target_rels)
                loss = loss_h + loss_r
                
                train_loss += loss_h.item() + loss_r.item()

                loss.backward()
                opt.step()

            train_loss /= n_train_batches

            parser.eval() #Crucial! Toggles dropout effects throughout network...
            dev_loss = 0
            arc_accuracy = 0
            rel_accuracy = 0
            for b in range(n_dev_batches):
                words, pos, sent_lens, target_heads, target_rels = next(dev_loader)

                S, L, head_preds = parser(words, pos, sent_lens)
                sys.exit()

                loss_h = loss_heads(S, target_heads)
                loss_r = loss_rels(L, target_rels)
                dev_loss = loss_h.item() + loss_r.item()

                head_accuracy = accuracy_heads(head_preds, target_heads)
                rel_accuracy = accuracy_rels(L, target_rels)

            dev_loss /= n_dev_batches

            #print('Training Loss:: Heads: {:.3f}\t Rels: {:.3f}\t Dev Accuracy:: Heads: {:.3f}\t Rels: {:.3f}\t'.format(
            #    arc_loss, rel_loss, arc_accuracy, rel_accuracy))
            print('Epoch:: {:.3f}\t [Training Loss:: {:.3f}\t Dev Loss:: {:.3f}]\t [Dev Accuracy:: Heads: {:.3f}\t Rels: {:.3f}]'.format(
                e, train_loss, dev_loss, head_accuracy, rel_accuracy))


            #LAS = #this is the number items where BOTH head and relation correctly predicted
            #if LAS > prev_best:
            #    earlystop_counter = 0 #reset
            #    prev_best = LAS
            #else:
            #    earlystop_counter += 1
            #    if earlystop_counter >= 5:
            #        print('LAS has not improved for 5 consecutive epochs, stopping after {:} epochs'.format(e))
            #        break

        #End of epoch loop

        #Save weights
        if not os.path.isdir(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        torch.save(parser.state_dict(), '%s-%s' % (WEIGHTS_PATH, MODEL_NAME))

    elif test:
        pass

#Eventually, probably should pull loss functions out into a utilities file
def loss_heads(S, target_heads, pad_idx = -1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)

    '''
    #print(target_heads.shape)
    #T1 = F.cross_entropy(S.permute(0,2,1), Variable(target_heads), ignore_index=pad_idx)
    #T2 = F.cross_entropy(S.permute(0,2,1), Variable(target_heads), ignore_index=pad_idx, reduction='none')
    #print(T1)
    #print(T2)
    #print(T2.size())
    #sys.exit()

    #For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S.permute(0,2,1), Variable(target_heads), ignore_index=pad_idx)

def loss_rels(L, target_rels, pad_idx = -1):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''
    return F.cross_entropy(L.permute(0,2,1), Variable(target_rels), ignore_index=pad_idx)

def accuracy_heads(head_preds, target_heads):
    #Incoming heads will have been predicted via heuristic

    batch_size = target_heads.shape[0]
    n_correct = head_preds.eq(target_heads).sum()

    return n_correct.item() / batch_size #XXX Wrong

def accuracy_rels(L, target_rels):
    #As of now, we predict relations by a simple argmax
    
    #rels should already be a LongTensor thanks to chunk_to_batch

    batch_size = target_rels.shape[0]
    #rel_preds = L.argmax(2).type(torch.LongTensor) #Must be LongTensor to do eq comparison with target
    rel_preds = L.argmax(2).long() #Must be LongTensor to do eq comparison with target
    n_correct = rel_preds.eq(target_rels).sum()


    #NOTE LAS is measured by counting the number of (head, relation) PAIRS that were correctly predicted

    return n_correct.item() / batch_size #XXX Wrong



def parse(head_preds, L):
    pass

    

if __name__ == '__main__':
    main()

