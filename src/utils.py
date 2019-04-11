import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

import data_utils

def attachment_scoring(
        head_preds=None, 
        rel_preds=None, 
        head_targets=None, 
        rel_targets=None, 
        sent_lens=None, 
        include_root=False, 
        keep_dim=False):
    '''
        input:
            head_preds - (b, l), (-1)-padded
            rel_preds -  (b, l), (-1)-padded
            head_targets - (-1)-padded (b, l) tensor of ints
            rel_targets - (-1)-padded (b, l) tensor of ints
            keep_dim - do not average across batch

        returns:
            UAS - average number of correct head predictions
            LAS - average number of correct relation predictions
    '''

    sent_lens = sent_lens.view(-1, 1)
    total_words = sent_lens.sum().float()
    b, l = head_preds.shape

    # To ensure padding values do not contribute to score in the .eq() calls
    head_preds = torch.where(
            head_targets != -1,
            head_preds,
            torch.zeros(head_preds.shape).long())
    rel_preds = torch.where(
            rel_targets != -1,
            rel_preds,
            torch.zeros(rel_preds.shape).long())

    correct_heads = head_preds.eq(head_targets).float()
    correct_rels = rel_preds.eq(rel_targets).float()

    UAS_correct = correct_heads.sum(1, True) # (b,l) -> (b,1)
    UAS_correct = UAS_correct if include_root else UAS_correct - 1
    if not keep_dim:
        UAS_correct = UAS_correct.sum() # (b,l) -> (1)
        UAS = UAS_correct / (total_words if include_root else total_words - 1)
    else:
        UAS = UAS_correct / (sent_lens if include_root else sent_lens - 1)

    LAS_correct = (correct_heads * correct_rels).sum(1, True)
    LAS_correct = LAS_correct if include_root else LAS_correct - 1
    if not keep_dim:
        LAS_correct = (correct_heads * correct_rels).sum()
        LAS = LAS_correct / (total_words if include_root else total_words - 1)
    else:
        LAS = LAS_correct / (sent_lens if include_root else sent_lens - 1)

    return {'UAS': UAS,
            'LAS': LAS, 
            'total_words' : total_words, 
            'UAS_correct' : UAS_correct,
            'LAS_correct' : LAS_correct}


def average_hiddens(hiddens, sent_lens):
    '''
        inputs:
            hiddens - tensor w/ shape (b, l, d)
            sent_lens - 1-D (b) tensor of sentence lengths

        outputs:
            averaged_hiddens -  (b, d) tensor
    '''

    #NOTE WE ARE ASSUMING PAD VALUES ARE 0 IN THIS SUM (NEED TO DOUBLE CHECK)
    averaged_hiddens = hiddens.sum(dim=1) # (b,l,2*h_size) -> (b,2*h_size)

    d = sent_lens.device
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

