import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict, Counter

from memory_profiler import profile

import string
import random

import sys


UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'


'''
   Things that still need doing:
   1. Shuffling the data
   2. A way of splitting to train/dev/test
'''

def get_dataset_sdp(conllu_file, training=False):
    sents_list = conllu_to_sents(conllu_file)

    sents_list, word_counts = filter_and_count(sents_list, filter_single=training)

    x2i_maps, i2x_maps = build_dicts(sents_list)
    
    sents_list = numericalize(sents_list, x2i_maps)
    
    return get_train_dev_test(sents_list), x2i_maps, i2x_maps, word_counts


def get_dataset_ss(paranmt_file, x2i_maps=None):
    sents_list = txt_to_sents(paranmt_file)

    sents_list = numericalize_ss(sents_list, x2i_maps)

    return sents_list


#TODO Possible edit required: EelcovdW's implementation
# uses chunks of shuffled INDICES rather than chunks of the
# data list itself; this may be a more convenient/efficient
# implementation once I get to the point of SHUFFLING the data
def sdp_data_loader(data, b_size):

    '''NOTE We pass the entirety of data_list as input to this function,
    which seems to not really make use of the space-efficient pattern of
    a generator. Should consider some kind of refactor here if time allows it.
    '''
    idx = list(range(len(data)))
    while True:
        shuffle(idx) # In-place shuffle
        for chunk in idx_chunks(idx, b_size):
            batch = [data[i] for i in chunk]
            yield prepare_batch_sdp(batch)


def ss_data_loader(data, b_size):
    '''
        inputs:
            data - the full Python list of pairs of numericalized sentences (np arrays)
            b_size - batch size

        yields:
            batch - list of selected data points (UNPREPARED)
    '''

    idx = list(range(len(data)))
    while True:
        shuffle(idx)
        for chunk in idx_chunks(idx, b_size):
            batch = [data[i] for i in chunk]
            yield batch
            #yield prepare_batch_ss(batch) OLD CODE


def idx_chunks(idx, chunk_size):
    for i in range(0, len(idx), chunk_size):
        yield idx[i:i+chunk_size]


def word_dropout(words, w2i=None, i2w=None, counts=None, lens=None, alpha=40):
    '''
       Words coming in as tensor, should be (b,l)
    '''
    for i, s in enumerate(words):
        for j in range(1, lens[i]): # Skip root token
            p = -1
            c = counts[ i2w[s[j]] ]
            p = alpha / (c + alpha) # Dropout probability
            if random.random() <= p:
                words[i,j] = int(w2i[UNK_TOKEN])


def prepare_batch_sdp(chunk):
    '''
    Transform a batch from a np array of sentences
    into several tensors to use in training
    '''

    batch_size = len(chunk)
    chunk_sorted = sorted(chunk, key = lambda s: s.shape[0], reverse=True)
    sent_lens = [s.shape[0] for s in chunk_sorted] # Keep in mind, these lengths include ROOT token in each sent
    length_longest = sent_lens[0]

    words = torch.zeros((batch_size, length_longest)).long()
    pos = torch.zeros((batch_size, length_longest)).long()
    heads = torch.Tensor(batch_size, length_longest).fill_(-1).long()
    rels = torch.Tensor(batch_size, length_longest).fill_(-1).long()

    for i, s in enumerate(chunk_sorted):
        for j, _ in enumerate(s):
            '''
            Casting as ints because for some stupid reason
            you cannot set a value in torch long tensor using
            numpy's 64 bit ints
            '''
            words[i,j] = int(s[j,0])
            pos[i,j] = int(s[j,1])
            heads[i,j] = int(s[j,2])
            rels[i,j] =  int(s[j,3])

    return words, pos, sent_lens, heads, rels


def prepare_batch_ss(batch):
    '''
        inputs:
            batch - list of lists of 2 or 3 np arrays (each a sentence)

        outputs:
            words - 2-tuple of tensors, shape (b,l)
            pos - 2-tuple of tensors, shape (b, l)
            sent_lens - 2-tuple of lists, shape (b)
    '''
    batch_size = len(batch) # TODO But really, megabatch size, right?
    n_sents = len(batch[0]) # 0 chosen arbitrarily

    words = [[] for i in range(n_sents)]
    pos = [[] for i in range(n_sents)]

    batch_sorted = [[] for i in range(n_sents)]
    sent_lens = [[] for i in range(n_sents)]
    length_longest = [[] for i in range(n_sents)]
    for instance in batch:
        for i in range(n_sents):
            batch_sorted[i].append(instance[i])
    for i in range(n_sents):
        batch_sorted[i] = sorted(batch_sorted[i], key=lambda s_i: s_i.shape[0], reverse=True)
        sent_lens[i] = [s_i.shape[0] for s_i in batch_sorted[i]]
        length_longest[i] = sent_lens[i][0]

    #length_longest = max(sent_lens[0][0], max([s[1].shape[0] for s in batch]))

    for i in range(n_sents):
        words[i] = torch.zeros((batch_size, length_longest[i])).long()
        pos[i] = torch.zeros((batch_size, length_longest[i])).long()
    #w1 = torch.zeros((batch_size, length_longest)).long()
    #w2 = torch.zeros((batch_size, length_longest)).long()
    #p1 = torch.zeros((batch_size, length_longest)).long()
    #p2 = torch.zeros((batch_size, length_longest)).long()

    for i in batch_srted
        for j, _ in enumerate(s):
            for k in enumerate

            words[i][j,k] = int(s_i[k,0])
            pos[i][j,k] = int(s_i[k,1])

    # for i, (s1, s2) in enumerate(batch_sorted):
    #     for j, _ in enumerate s1:
    #         w1[i,j] = int(s1[j,0])
    #         p1[i,j] = int(s1[j,1])
    #     for j, _ in enumerate s2:
    #         w2[i,j] = int(s2[j,0])
    #         p2[i,j] = int(s2[j,1])

    return words, pos, sent_lens 


def conllu_to_sents(f: str):
    '''
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)

    '''

    mask = [1, 4, 6, 7]  # [word, pos, head, rel]

    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sents_list = []
    sent_start = 0
    for sent_end in split_points: # Assumes the final line is '\n'
        sents_list.append(lines[sent_start: sent_end])
        sent_start = sent_end + 1 # Skipping the line break

    for i, s in enumerate(sents_list):
        s_split = [line.split('\t') for line in s]
        sents_list[i] = np.array(s_split)[:, mask]

    return sents_list


def txt_to_sents(f: str):
    '''
        inputs:
            f - name of sentences/paraphrases dataset txt file

        outputs:
            sents_list - a list of pairs (tuples) of sentences and their
                         paraphrases
    '''

    with open(f, 'r') as txt_file:
        lines = txt_file.readlines()

    sents_list = []
    for line in lines:
        sents = line.split('\t')
        s1 = sents[0].strip().split(' ')
        s2 = sents[1].strip().split(' ')
        sents_list.append( (s1,s2) )

    return sents_list


def build_dicts(sents_list):
    words, pos, rel = set(), set(), set()
    for s in sents_list:
        for line in s:
            words.add(line[0].lower())
            pos.add(line[1])
            rel.add(line[3])

    w2i = defaultdict(lambda : len(w2i))
    p2i = defaultdict(lambda : len(p2i))
    r2i = defaultdict(lambda : len(r2i))
    i2w, i2p, i2r = dict(), dict(), dict()

    #Crucial that PAD_TOKEN map to 0 so that chunk_to_batch() definition correct
    i2w[w2i[PAD_TOKEN]] = PAD_TOKEN
    i2p[p2i[PAD_TOKEN]] = PAD_TOKEN

    i2w[w2i[UNK_TOKEN]] = UNK_TOKEN
    i2p[p2i[UNK_TOKEN]] = UNK_TOKEN

    i2w[w2i[ROOT_TOKEN]] = ROOT_TOKEN
    i2p[p2i[ROOT_TOKEN]] = ROOT_TOKEN

    for w in words:
        i2w[w2i[w]] = w
    for p in pos:
        i2p[p2i[p]] = p
    for r in rel:
        i2r[r2i[r]] = r

    x2i_maps = {'word' : dict(w2i), 'pos' : dict(p2i), 'rel' : dict(r2i)}
    i2x_maps = {'word' : i2w, 'pos' : i2p, 'rel' : i2r}

    return x2i_maps, i2x_maps


def numericalize(sents_list, x2i_maps):
    w2i = x2i_maps['word']
    p2i = x2i_maps['pos']
    r2i = x2i_maps['rel']

    sents_numericalized = []
    for s in sents_list:
        new_shape = (s.shape[0] + 1, s.shape[1])

        new_s = np.zeros(new_shape, dtype=int) # Making room for ROOT_TOKEN
        new_s[0,:] = w2i[ROOT_TOKEN], p2i[ROOT_TOKEN], -1, -1 # -1s here become crucial for attachment scoring

        for i in range(s.shape[0]):
            new_s[i+1,0] = w2i.get(s[i,0].lower(), w2i[UNK_TOKEN])
            new_s[i+1,1] = p2i.get(s[i,1], p2i[UNK_TOKEN])
            new_s[i+1,2] = int(s[i,2]) # Head idx
            new_s[i+1,3] = r2i[s[i,3]]

        sents_numericalized.append(new_s)

    return sents_numericalized


def numericalize_ss(sents_list, x2i_maps):
    w2i = x2i_maps['word']
    p2i = x2i_maps['pos']

    sents_numericalized = []
    for s1, s2 in sents_list:
        new_s1 = np.zeros((len(s1), 2), dtype=int)
        new_s2 = np.zeros((len(s2), 2), dtype=int)

        for i in range(len(s1)):
            new_s1[i,0] = w2i.get(s1[i].lower(), w2i[UNK_TOKEN])
            new_s1[i,1] = p2i[UNK_TOKEN]
        for i in range(len(s2)):
            new_s2[i,0] = w2i.get(s2[i].lower(), w2i[UNK_TOKEN])
            new_s2[i,1] = p2i[UNK_TOKEN]

        sents_numericalized.append( (new_s1, new_s2) )

    return sents_numericalized


# From https://github.com/EelcovdW/Biaffine-Parser/blob/master/data_utils.py
def filter_and_count(sentences, filter_single=True):
    """
    Applies a series of filter to each word in each sentence. Filters
    are applied in this order:
    - replace urls with an <url> tag.
    - replace a string of more than 2 punctuations with a <punct> tag.
    - replace strings that contain digits with a <num> tag.
    - if filter_single, replace words that only occur once with UNK_TOKEN.
      This step is useful when parsiline training data, to make sure the UNK_TOKEN
      in the word embeddings gets trained.
    Args:
        sentences: list of sentences, from parse_conllu.
        filter_single: boolean, if true replace words that occur once with UNK_TOKEN.
    Returns: List of sentences with words filtered.
    """
    filtered = []
    word_counts = get_word_counts(sentences)
    one_words = set([w for w, c in word_counts.items() if c == 1])
    for i, sentence in enumerate(sentences):
        for j, line in enumerate(sentence):
            word = line[0]
            if is_url(word):
                word = '<url>'
            elif is_long_punctuation(word):
                word = '<punct>'
            elif has_digits(word):
                word = '<num>'
            elif filter_single and word.lower() in one_words:
                word = UNK_TOKEN

        filtered.append(sentence)

    return filtered, word_counts

def get_word_counts(sentences):
    """
    Create a Counter of all words in sentences, in lowercase.
    Args:
        sentences: List of sentences, from parse_conllu.
    Returns: Counter with word: count.
    """
    words = [line[0].lower() for sentence in sentences for line in sentence]
    return Counter(words)


def is_url(word):
    """
    Lazy check if a word is an url. True if word contains all of {':' '/' '.'}.
    """
    return bool(set('./:').issubset(word))


def is_long_punctuation(word):
    """
    True if word is longer than 2 and only contains interpunction.
    """
    return bool(len(word) > 2 and set(string.punctuation).issuperset(word))


def has_digits(word):
    """
    True if word contains digits.
    """
    return bool(set(string.digits).intersection(word))


def get_train_dev_test(data_list):
    n_samples = len(data_list)

    #XXX Incredibly stupid way of splitting data
    x = int(.8 * n_samples)
    y = int(.9 * n_samples)

    return { 'train': data_list[:x], 'dev': data_list[x:y], 'test': data_list[y:] }

# End of https://github.com/EelcovdW/Biaffine-Parser/blob/master/data_utils.py

#def testing():
#    #sents_list = conllu_to_sents('/Users/dylanfinkbeiner/Desktop/stanford-parser-full-2018-10-17/treebank.conllu')
#
#    #dict2, _ =  build_dicts(sents_list)
#
#    #numd = numericalize(sents_list, dict2)
#
#    _, x2i, i2x, _  = get_dataset_sdp('../data/tenpercentsample.conllu')
#    f = '../data/para_sample.txt'
#
#    sents_before, sents_after = get_dataset_ss(f, x2i)
#
#    print(sents_before[0])
#    print(sents_after[0])
#    print(sents_before[-1])
#    print(sents_after[-1])
#
#
#
#if __name__ == '__main__':
#    testing()
