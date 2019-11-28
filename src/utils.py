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
        keep_dim=False,
        original_sentence=None,
        dep_of_unk=None):
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
    rel_preds = rel_preds.to(device)
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

    total_arc_errors = -1
    total_rel_errors = -1
    total_unks = -1
    total_dep_of_unks = -1
    arc_unk_errors = -1
    rel_unk_errors = -1
    dep_of_unk_arc_errors = -1
    dep_of_unk_rel_errors = -1
    if type(original_sentence) != type(None):
        # We want 0's in this matrix to correspond to cases where arc was predicted correctly but rel was not
        interesting_rels = torch.where(
                correct_arcs == 1,
                correct_rels,
                torch.ones(correct_rels.shape).to(device))

        unks = (original_sentence == 1).float() # UNK token integer encoding is 1
        arc_unk_errors = torch.where(
                correct_arcs == 0,
                unks,
                torch.zeros(unks.shape).to(device))
        rel_unk_errors = torch.where(
                interesting_rels == 0,
                unks,
                torch.zeros(unks.shape).to(device))

        dep_of_unk_arc_errors = torch.where(
                correct_arcs == 0,
                dep_of_unk,
                torch.zeros(unks.shape).to(device))
        dep_of_unk_rel_errors = torch.where(
                interesting_rels == 0,
                dep_of_unk,
                torch.zeros(unks.shape).to(device))


        total_rel_errors = (interesting_rels == 0.).sum()
        total_arc_errors = (correct_arcs == 0.).sum()
        total_unks = unks.sum()
        total_dep_of_unks = dep_of_unk.sum()
        arc_unk_errors = arc_unk_errors.sum()
        rel_unk_errors = rel_unk_errors.sum()
        dep_of_unk_arc_errors = dep_of_unk_arc_errors.sum()
        dep_of_unk_rel_errors = dep_of_unk_rel_errors.sum() 



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
            'LAS_correct' : LAS_correct,
            'total_rel_errors' : total_rel_errors,
            'total_arc_errors' : total_arc_errors,
            'total_unks' : total_unks,
            'total_dep_of_unks' : total_dep_of_unks,
            'arc_unk_errors' : arc_unk_errors,
            'rel_unk_errors' : rel_unk_errors,
            'dep_of_unk_arc_errors' : dep_of_unk_arc_errors,
            'dep_of_unk_rel_errors' : dep_of_unk_rel_errors}


def average_hiddens(hiddens, sent_lens=None, sum_f_b=False):
    #NOTE WE ARE ASSUMING PAD VALUES ARE 0 IN THIS SUM
    averaged_hiddens = hiddens.sum(dim=1) # (b,l,2*h_size) -> (b,2*h_size)
    sent_lens = sent_lens.view(-1, 1).float() # (b, 1)
    averaged_hiddens /= sent_lens

    if sum_f_b:
        halfway = averaged_hiddens.shape[1] // 2
        averaged_hiddens = averaged_hiddens[:,:halfway] + averaged_hiddens[:,halfway:]
        averaged_hiddens /= 2

    return averaged_hiddens


def predict_relations(S_rel):
    rel_preds = S_rel.cpu().argmax(2).long()
    return rel_preds


def sts_scoring(predictions, targets) -> float:
    R, p_value = pearsonr(predictions, targets)
    return R


def predict_sts_score(sem_h1, sem_h2, conventional_range=False):
    sims = F.cosine_similarity(sem_h1, sem_h2, dim=-1).flatten()

    # Scale into 0-5 range, per SemEval STS task conventions
    if conventional_range:
        sims += 1
        sims *= 2.5

    return sims.tolist()


def word_dropout(words, w2i=None, i2w=None, counts=None, lens=None, rate=None, style='freq'):
    mask = torch.ones(words.shape)
    if rate > 0.:
        if style == 'unif':
            assert(rate <= 1. and rate >= 0)
        unk_i = int(w2i[UNK_TOKEN])
        dropped = torch.LongTensor(words)
        for i, sentence in enumerate(words):
            for j in range(1, lens[i]): # Skip root token (assumes lens include root)
                p = -1
                if style == 'freq':
                    c = counts[ i2w[sentence[j].item()] ]
                    p = rate / (c + rate) # Dropout probability
                elif style == 'unif':
                    p = rate
                if random.random() <= p:
                    dropped[i,j] = unk_i
                    mask[i,j] = 0
        return dropped, mask
    else:
        return words, mask


def pos_dropout(pos, lens=None, p2i=None, p=None):
    if type(pos) == type(None):
        return None, None
    else:
        pos = pos.to('cpu')
        mask = torch.ones(pos.shape)
        if p > 0.:
            assert(p <= 1. and p >= 0)
            unk_i = int(p2i[UNK_TOKEN])
            dropped = torch.LongTensor(pos)
            for i, sentence in enumerate(pos):
                for j in range(1, lens[i]):
                    if random.random() <= p:
                        dropped[i,j] = unk_i
                        mask[i,j] = 0
            return dropped, mask
        else:
            return pos, mask
