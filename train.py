import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

from parser import BiaffineParser
from ptb_dataset import get_dataset, custom_data_loader

from math import ceil

NUM_EPOCHS = 100
BATCH_SIZE = 10

WEIGHTS_PATH = '../weights'
MODEL_NAME = 'newmodel'

CONLLU_FILE = './data/treebank.conllu'

train = True


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
        dataset, word_vsize, pos_vsize, rel_vsize  = get_dataset(CONLLU_FILE)
        loader = custom_data_loader(dataset, BATCH_SIZE) 
        parser = BiaffineParser(
                word_vocab_size = word_vsize, 
                pos_vocab_size = pos_vsize,
                num_relations = rel_vsize)

        n_batches = ceil(len(dataset) / BATCH_SIZE)

        #Optimizer
        opt = Adam(parser.parameters(), lr = 1e-2)

        for e in range(NUM_EPOCHS):

            parser.train()
            for b in range(n_batches):
                opt.zero_grad()

                words, pos, sent_lens, heads, rels = next(loader)

                S, L = parser(words, pos, sent_lens, train=True)

                #Calculate losses
                loss = loss_heads(S, heads)
                loss += loss_rels(L, rels)

                print("Loss")
                loss.backward()
                opt.step()

        #Save weights
        if not os.path.isdir(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        torch.save(parser.state_dict(), '%s-%s' % (WEIGHTS_PATH, MODEL_NAME))

    elif test:
        pass

def train_test_split(data):
    pass

#Eventually, probably should pull loss functions out into a utilities file
def loss_heads(S, heads):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)

    '''
    Y_heads = Variable(heads)

    return F.cross_entropy(S, Y_heads)

def loss_rels(L, rels):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''

    Y_labels = Variable(rels)

    print("Y_labels shape: ", Y_labels.shape)

    #XXX As of now, permuting axes 1 and 2 to represent "transpose", not sure if this is right
    return F.cross_entropy(L.permute(0,2,1), Y_labels)

if __name__ == '__main__':
    main()

