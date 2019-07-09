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


def word_dropout(words, w2i=None, i2w=None, counts=None, lens=None, alpha=None):
    mask = torch.ones(words.shape)
    if alpha > 0.:
        unk_i = int(w2i[UNK_TOKEN])
        dropped = torch.LongTensor(words)
        for i, sentence in enumerate(words):
            for j in range(1, lens[i]): # Skip root token (assumes lens include root)
                p = -1
                c = counts[ i2w[sentence[j].item()] ]
                p = alpha / (c + alpha) # Dropout probability
                if random.random() <= p:
                    dropped[i,j] = unk_i
                    mask[i,j] = 0
        return dropped, mask
    else:
        return words, mask
