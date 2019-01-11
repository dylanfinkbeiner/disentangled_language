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

        print(f'There are {len(train_data)} training examples.')
        print(f'There are {len(dev_data)} validation examples.')

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

        #earlystop_counter = 0
        #prev_best = 0

        print('Starting train loop.')
        for e in range(NUM_EPOCHS):

            parser.train()
            train_loss = 0
            for b in range(n_train_batches):
                opt.zero_grad()

                words, pos, sent_lens, head_targets, rel_targets = next(train_loader)

                S, L, _ = parser(words, pos, sent_lens)

                loss_h = loss_heads(S, head_targets)
                loss_r = loss_rels(L, rel_targets)
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
                words, pos, sent_lens, head_targets, rel_targets = next(dev_loader)

                S, L, head_preds = parser(words, pos, sent_lens)
                rel_preds = predict_relations(L, head_preds)

                loss_h = loss_heads(S, head_targets)
                loss_r = loss_rels(L, rel_targets)
                dev_loss = loss_h.item() + loss_r.item()

                #head_accuracy = accuracy_heads(head_preds, head_targets)
                #rel_accuracy = accuracy_rels(L, rel_targets)

                UAS, LAS = attachment_scoring(
                        head_preds, 
                        rel_preds, 
                        head_targets,
                        rel_targets,
                        sent_lens)

            dev_loss /= n_dev_batches

            #print('Training Loss:: Heads: {:.3f}\t Rels: {:.3f}\t Dev Accuracy:: Heads: {:.3f}\t Rels: {:.3f}\t'.format(
            #    arc_loss, rel_loss, arc_accuracy, rel_accuracy))
            #print('Epoch:: {:.3f}\t [Training Loss:: {:.3f}\t Dev Loss:: {:.3f}]\t [Dev Accuracy:: Heads: {:.3f}\t Rels: {:.3f}]'.format(
            #    e, train_loss, dev_loss, head_accuracy, rel_accuracy))
            print('Epoch:: {:}\t [Training Loss:: {:.3f}\t Dev Loss:: {:.3f}]\t [Dev Accuracy:: UAS: {:.3f}\t LAS: {:.3f}]'.format(
                e, train_loss, dev_loss, UAS, LAS))


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
def loss_heads(S, head_targets, pad_idx = -1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)

    '''
    #print(head_targets.shape)
    #T1 = F.cross_entropy(S.permute(0,2,1), Variable(head_targets), ignore_index=pad_idx)
    #T2 = F.cross_entropy(S.permute(0,2,1), Variable(head_targets), ignore_index=pad_idx, reduction='none')
    #print(T1)
    #print(T2)
    #print(T2.size())
    #sys.exit()

    #For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S.permute(0,2,1), Variable(head_targets), ignore_index=pad_idx)

def loss_rels(L, rel_targets, pad_idx = -1):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''
    return F.cross_entropy(L.permute(0,2,1), Variable(rel_targets), ignore_index=pad_idx)

def predict_relations(L, head_preds):
    '''
    args
        L::Tensor - label logits with shape (b, l, num_rels)

    returns 
        rel_preds - shape (b, l)
    '''

    rel_preds = L.argmax(2).long()
    return rel_preds


#def accuracy_heads(head_preds, head_targets):
#    #Incoming heads will have been predicted via heuristic
#
#    batch_size = head_targets.shape[0]
#    n_correct = head_preds.eq(head_targets).sum()
#
#    return n_correct.item() / batch_size #XXX Wrong
#
#def accuracy_rels(L, head_preds, rel_targets):
#    #As of now, we predict relations by a simple argmax
#    
#    #rels should already be a LongTensor thanks to chunk_to_batch
#
#    batch_size = rel_targets.shape[0]
#    #rel_preds = L.argmax(2).type(torch.LongTensor) #Must be LongTensor to do eq comparison with target
#    rel_preds = L.argmax(2).long() #Must be LongTensor to do eq comparison with target
#    n_correct = rel_preds.eq(rel_targets).sum()
#
#
#    #NOTE LAS is measured by counting the number of (head, relation) PAIRS that were correctly predicted
#
#    return n_correct.item() / batch_size #XXX Wrong

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

    # This way we can be sure padding values do not contribute to score
    head_preds = torch.where(head_targets != -1, head_preds, torch.zeros(head_preds.shape).long())
    rel_preds = torch.where(rel_targets != -1, rel_preds, torch.zeros(rel_preds.shape).long())

    # Tensors with 1s in locations of correct predictions 
    #NOTE this could be optimized later to avoid sparse matrices
    correct_heads = head_preds.eq(head_targets).float()
    #print("Rel preds is, ", rel_preds[0:3])
    correct_rels = rel_preds.eq(rel_targets).float()
    #print("Target rels is, ", rel_targets[0:3])

    UAS = correct_heads.sum(1, True)
    #print("Correct heads sum:", UAS)
    UAS /= sent_lens
    UAS = UAS.sum() / b
    #print("Finished UAS: ", UAS)

    #print(correct_heads)
    #print(correct_rels)
    #print("correct_rels", correct_rels[0:3])
    LAS = (correct_heads * correct_rels).sum(1, True)
    #print("Correct heads eq correct rels", LAS)
    LAS /= sent_lens
    LAS = LAS.sum() / b
    #print("Finished LAS:", LAS)

    return UAS, LAS

if __name__ == '__main__':
    main()

