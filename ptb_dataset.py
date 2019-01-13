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

def get_dataset(conllu_file, training=False):
    sents_list = conllu_to_sents(conllu_file)

    x2num_maps, num2x_maps = build_dicts(sents_list)
    
    sents_list, word_counts = filter_and_count(sents_list, filter_single=training)
    
    sent_list = numericalize(sents_list, x2num_maps)
    
    return get_train_dev_test(sent_list), x2num_maps, num2x_maps, word_counts


def get_train_dev_test(data_list):
    n_samples = len(data_list)

    #XXX Incredibly stupid way of splitting data
    x = int(.8 * n_samples)
    y = int(.9 * n_samples)

    return {'train': data_list[:x], 'dev': data_list[x:y], 'test': data_list[y:] }


#TODO Possible edit required: EelcovdW's implementation
# uses chunks of shuffled INDICES rather than chunks of the
# data list itself; this may be a more convenient/efficient
# implementation once I get to the point of SHUFFLING the data
def custom_data_loader(data, b_size, word2num, num2word, word_counts):

    '''NOTE We pass the entirety of data_list as input to this function,
    which seems to not really make use of the space-efficient pattern of
    a generator. Should consider some kind of refactor here if time allows it.
    '''
    chunk_generator = (data[i:i+b_size] for i in range(0, len(data, b_size)))

    while True:
        for chunk in chunk_generator:
            yield chunk_to_batch(chunk, word2num, num2word, word_counts)

    #while True:
    #    for chunk in chunk_generator(data, batch_size):
    #        yield chunk_to_batch(chunk, word2num, num2word, word_counts)


def chunk_generator(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]


def chunk_to_batch(chunk, word2num, num2word, word_counts):
    '''
    Transform a chunk (list) of sentences
    into a batch for training the parser
    in the form of five LongTensors representing
    the sentences as words, pos, arcs, deprels

    requires PADDING
    '''

    alpha = 40 # NOTE This is not a great place for this variable.

    batch_size = len(chunk)
    chunk_sorted = sorted(chunk, key = lambda s: s.shape[0], reverse=True)
    sent_lens = [np.shape(s)[0] for s in chunk_sorted] # Keep in mind, these lengths include ROOT token in each sent
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
            p = -1
            if j > 0: # First token in a sentence is ROOT_TOKEN
                c = word_counts[num2word[s[j,0]]]
                p = alpha / (c + alpha) # Dropout probability

            words[i,j] = int(s[j,0]) if random.random() > p else int(word2num[UNK_TOKEN])
            pos[i,j] = int(s[j,1])
            heads[i,j] = int(s[j,2])
            rels[i,j] =  int(s[j,3])


    return words, pos, sent_lens, heads, rels


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
        sent_start = sent_end + 1

    for i, s in enumerate(sents_list):
        s_split = [line.split('\t') for line in s]
        sents_list[i] = np.array(s_split)[:, mask]

    return sents_list


def build_dicts(sents_list):
    words, pos, rel = set(), set(), set()
    for s in sents_list:
        for line in s:
            words.add(line[0].lower())
            pos.add(line[1])
            rel.add(line[3])

    word2num = defaultdict(lambda : len(word2num))
    pos2num = defaultdict(lambda : len(pos2num))
    rel2num = defaultdict(lambda : len(rel2num))
    num2word, num2pos, num2rel = dict(), dict(), dict()

    #Crucial that PAD_TOKEN map to 0 so that chunk_to_batch() definition correct
    num2word[word2num[PAD_TOKEN]] = PAD_TOKEN
    num2pos[pos2num[PAD_TOKEN]] = PAD_TOKEN

    num2word[word2num[UNK_TOKEN]] = UNK_TOKEN

    num2word[word2num[ROOT_TOKEN]] = ROOT_TOKEN
    num2pos[pos2num[ROOT_TOKEN]] = ROOT_TOKEN

    for w in words:
        num2word[word2num[w]] = w
    for p in pos:
        num2pos[pos2num[p]] = p
    for r in rel:
        num2rel[rel2num[r]] = r

    x2num_maps = {'word' : word2num, 'pos' : pos2num, 'rel' : rel2num}
    num2x_maps = {'word' : num2word, 'pos' : num2pos, 'rel' : num2rel}

    return x2num_maps, num2x_maps


def numericalize(sents_list, x2num_maps):
    word2num = x2num_maps['word']
    pos2num = x2num_maps['pos']
    rel2num = x2num_maps['rel']

    sents_numericalized = []
    for s in sents_list:
        new_shape = (s.shape[0] + 1, s.shape[1])
        new_s = np.zeros(new_shape, dtype=int) # Making room for ROOT_TOKEN
        new_s[0,:] = word2num[ROOT_TOKEN], pos2num[ROOT_TOKEN], -1, -1 # -1s here become crucial for attachment scoring

        for i in range(s.shape[0]):
            new_s[i+1,0] = word2num.get(s[i,0].lower(), word2num[UNK_TOKEN])
            new_s[i+1,1] = pos2num.get(s[i,1], pos2num[UNK_TOKEN])
            new_s[i+1,2] = int(s[i,2]) # Head idx
            new_s[i+1,3] = rel2num[s[i,3]]

        sents_numericalized.append(new_s)

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

# End of https://github.com/EelcovdW/Biaffine-Parser/blob/master/data_utils.py

#def testing():
#    #sents_list = conllu_to_sents('/Users/dylanfinkbeiner/Desktop/stanford-parser-full-2018-10-17/treebank.conllu')
#
#    #dict2, _ =  build_dicts(sents_list)
#
#    #numd = numericalize(sents_list, dict2)
#
#    data_list = get_dataset('/Users/dylanfinkbeiner/Desktop/stanford-parser-full-2018-10-17/treebank.conllu')
#
#    loader = custom_data_loader(data_list, 10)
#
#    print(next(loader))
#
#
#
#if __name__ == '__main__':
#    testing()
