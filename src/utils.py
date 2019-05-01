import random
from random import shuffle

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

import data_utils

UNK_TOKEN = '<unk>'

def attachment_scoring(
        arc_preds=None, 
        rel_preds=None, 
        arc_targets=None, 
        rel_targets=None, 
        sent_lens=None, 
        include_root=False, 
        keep_dim=False):
    '''
        input:
            arc_preds - (b, l), (-1)-padded
            rel_preds -  (b, l), (-1)-padded
            arc_targets - (-1)-padded (b, l) tensor of ints
            rel_targets - (-1)-padded (b, l) tensor of ints
            keep_dim - do not average across batch

        returns:
            UAS - average number of correct arc predictions
            LAS - average number of correct relation predictions
    '''

    device = arc_preds.device # Tensor chosen mostly arbitrarily
    arc_targets = arc_targets.to(device)
    rel_targets = rel_targets.to(device)
    sent_lens = sent_lens.to(device)

    # CRUCIAL to remember sentence lengths INCLUDE root token in their count
    sent_lens = sent_lens.view(-1, 1).float()
    sent_lens = sent_lens if include_root else sent_lens - 1
    total_words = sent_lens.sum().float()
    b, l = arc_preds.shape

    # CRUCIAL to remember that targets come with -1 padding in the ROOT position!
    # This means that the root predictions are NEVER counted towards correct #
    arc_preds = torch.where(
            arc_targets != -1,
            arc_preds,
            torch.zeros(arc_preds.shape).long().to(device))

    rel_preds = torch.where(
            rel_targets != -1,
            rel_preds,
            torch.zeros(rel_preds.shape).long().to(device))

    correct_arcs = arc_preds.eq(arc_targets).float()
    correct_rels = rel_preds.eq(rel_targets).float()

    UAS_correct = correct_arcs.sum(1, True) # (b,l) -> (b,1)
    UAS_correct = UAS_correct + 1 if include_root else UAS_correct
    if not keep_dim:
        UAS_correct = UAS_correct.sum() # (b,1) -> (1)
        UAS = UAS_correct / total_words
    else:
        UAS = UAS_correct / sent_lens

    LAS_correct = (correct_arcs * correct_rels).sum(1, True) # (b,l) -> (b,1)
    LAS_correct = LAS_correct + 1 if include_root else LAS_correct
    if not keep_dim:
        LAS_correct = LAS_correct.sum() # (b,1) -> (1)
        LAS = LAS_correct / total_words
    else:
        LAS = LAS_correct / sent_lens

    return {'UAS': UAS,
            'LAS': LAS, 
            'total_words' : total_words, 
            'UAS_correct' : UAS_correct,
            'LAS_correct' : LAS_correct}


def average_hiddens(hiddens, sent_lens=None):
    '''
        inputs:
            hiddens - tensor w/ shape (b, l, d)
            sent_lens - 1-D (b) tensor of sentence lengths

        outputs:
            averaged_hiddens -  (b, d) tensor
    '''

    #NOTE WE ARE ASSUMING PAD VALUES ARE 0 IN THIS SUM (NEED TO DOUBLE CHECK)
    averaged_hiddens = hiddens.sum(dim=1) # (b,l,2*h_size) -> (b,2*h_size)

    sent_lens = sent_lens.view(-1, 1).float() # (b, 1)

    averaged_hiddens /= sent_lens

    return averaged_hiddens


def predict_relations(S_rel):
    '''
        inputs:
            S_rel - label logits with shape (b, l, num_rels)

        outputs:
            rel_preds - shape (b, l)
    '''

    rel_preds = S_rel.cpu().argmax(2).long()
    
    return rel_preds


def sts_scoring(predictions, targets) -> float:
    r, _ = pearsonr(predictions, targets)
    return r


def predict_sts_score(h1, h2, h_size=None, syn_size=None, conventional_range=False):
    sem_h1 = torch.cat((h1[:,syn_size:h_size], h1[:,h_size+syn_size:]), dim=-1)
    sem_h2 = torch.cat((h2[:,syn_size:h_size], h2[:,h_size+syn_size:]), dim=-1)

    # Code for solving SemEval mystery  
    #sem_h1 = torch.randn(h1.shape)
    #sem_h2 = torch.randn(h2.shape)

    #breakpoint()

    sims = F.cosine_similarity(sem_h1, sem_h2, dim=-1).flatten()

    #breakpoint()

    # Scale into 0-5 range, per SemEval STS task conventions (commented out since R invariant to linear transformations)
    if conventional_range:
        sims += 1
        sims *= 2.5

    return sims.tolist()


def word_dropout(words, w2i=None, i2w=None, counts=None, lens=None, alpha=40):
    '''
        inputs:
            words - LongTensor, shape (b,l)
            w2i - word to index dict
            i2w - index to word dict
            counts - Counter object associating words to counts in corpus
            lens - lens of sentences (should be b of them)
            alpha - hyperparameter for dropout

        outputs:
            dropped - new LongTensor, shape (b,l)
    '''
    dropped = torch.LongTensor(words)

    for i, s in enumerate(words):
        for j in range(1, lens[i]): # Skip root token
            p = -1
            c = counts[ i2w[s[j].item()] ]
            p = alpha / (c + alpha) # Dropout probability
            if random.random() <= p:
                dropped[i,j] = int(w2i[UNK_TOKEN])
    
    return dropped
