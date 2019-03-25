import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import pickle

#from memory_profiler import profile

import string
import random
from random import shuffle

from nltk.parse import CoreNLPParser

import sys


UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'

CONLLU_MASK = [1, 4, 6, 7]  # [word, pos, head, rel]
CORENLP_URL = 'http://localhost:9000'


# Sections 2-21 for training, 22 for dev, 23 for test
def build_dataset_sdp(conllu_files=[]):
    '''
        inputs:
            conllu_files - a list of sorted strings, filenames of dependencies

        output:
            
    '''
    sents_list = []

    for f in conllu_files:
        sents_list.append(conllu_to_sents(f))

    if len(sents_list) != 24:
        print(f'Missing a conllu file? {len(sents_list)} files provided.')
        raise Exception

    for i, f in enumerate(sents_list):
        sents_list[i] = [s[:, CONLLU_MASK] for s in f]

    # "Standard" train/dev/split for PTB
    train_list = [s for f in sents_list[2:22] for s in f]
    dev_list = sents_list[22]
    test_list = sents_list[23]

    #sorted_sents = [i for i in sents_list if i.shape[0] <= 5]
    #for i in sorted_sents[:100]:
    #    print(i)
    #    print('\n')
    #exit()

    train_list, word_counts = filter_and_count(train_list, filter_single=True)
    dev_list, _ = filter_and_count(dev_list, filter_single=False)
    test_list, _ = filter_and_count(test_list, filter_single=False)

    x2i, i2x = build_dicts(train_list)
    
    train_list = numericalize_sdp(train_list, x2i)
    dev_list = numericalize_sdp(dev_list, x2i)
    test_list = numericalize_sdp(test_list, x2i)

    data_sdp = {
            'train': train_list,
            'dev': dev_list,
            'test': test_list
            }
    
    return data_sdp, x2i, i2x, word_counts


def build_dataset_ss(paranmt_file, x2i=None):
    sents_list = para_to_sents(paranmt_file)

    sents_list = numericalize_ss(sents_list, x2i)

    return sents_list


def sdp_data_loader(data, batch_size=1, shuffle_idx=False):

    '''NOTE We pass the entirety of data_list as input to this function,
    which seems to not really make use of the space-efficient pattern of
    a generator. Should consider some kind of refactor here if time allows it.
    '''
    idx = list(range(len(data)))
    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle
        for chunk in idx_chunks(idx, batch_size):
            batch = [data[i] for i in chunk]
            yield prepare_batch_sdp(batch)


def ss_data_loader(data, b_size):
    '''
        inputs:
            data - the full Python list of pairs of numericalized sentences (np arrays)
            b_size - batch size

        yields:
            chunk - list of indices representing a minibatch
    '''

    idx = list(range(len(data)))
    while True:
        shuffle(idx)
        for chunk in idx_chunks(idx, b_size):
            yield chunk


def idx_chunks(idx, chunk_size):
    for i in range(0, len(idx), chunk_size):
        yield idx[i:i+chunk_size]


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


def prepare_batch_sdp(batch):
    '''
        inputs:
            batch - 

        outputs:
            words - 
            pos -
            sent_lens - list of lengths (INCLUDES ROOT TOKEN)
            heads -
            rels -
    '''

    batch_size = len(batch)
    batch_sorted = sorted(batch, key = lambda s: s.shape[0], reverse=True)
    sent_lens = [s.shape[0] for s in batch_sorted] # Keep in mind, these lengths include ROOT token in each sentence
    length_longest = sent_lens[0]

    words = torch.zeros((batch_size, length_longest)).long()
    pos = torch.zeros((batch_size, length_longest)).long()
    heads = torch.Tensor(batch_size, length_longest).fill_(-1).long()
    rels = torch.Tensor(batch_size, length_longest).fill_(-1).long()

    for i, s in enumerate(batch_sorted):
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
            batch - batch as a list of numpy arrays representing sentences

        outputs:
            words - LongTensor, shape (b,l), padded with zeros
            pos - LongTensor, shape (b,l), padded with zeros
            sent_lens - list of sentence lengths (integers)
    '''

    batch_size = len(batch)

    sent_lens = [s.shape[0] for s in batch]
    length_longest = max(sent_lens)

    words = torch.zeros((batch_size, length_longest)).long()
    pos = torch.zeros((batch_size, length_longest)).long()

    for i, s in enumerate(batch):
        for j, _ in enumerate(s):
            '''
            Casting as ints because for some stupid reason
            you cannot set a value in torch long tensor using
            numpy's 64 bit ints
            '''
            words[i,j] = int(s[j,0])
            pos[i,j] = int(s[j,1])

    return words, pos, sent_lens


def conllu_to_sents(f: str):
    '''
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)
    '''


    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sents_list = []
    sent_start = 0
    for sent_end in split_points: # Assumes the final line is '\n'
        sents_list.append(lines[sent_start: sent_end])
        sent_start = sent_end + 1 # Skipping the line break

    for i, s in enumerate(sents_list):
        s_split = [line.rstrip().split('\t') for line in s]
        sents_list[i] = np.array(s_split)

    return sents_list


def para_to_sents(f: str):
    '''
        inputs:
            f - name of sentences/paraphrases dataset txt file

        outputs:
            sents_list - a list of pairs (tuples) of sentences and their
                         paraphrases
    '''

    # TODO Some kind of try/catch here if server connection fails?
    tagger = CoreNLPParser(url=f'{CORENLP_URL}', tagtype='pos')

    with open(f, 'r') as para_file:
        lines = para_file.readlines()

    sents_list = []
    for line in lines:
        sents = line.split('\t')
        s1 = sents[0].strip().split(' ')
        s2 = sents[1].strip().split(' ')
        s1 = np.array(tagger.tag(s1))
        s2 = np.array(tagger.tag(s2))
        sents_list.append( (s1,s2) )

    return sents_list


def build_dicts(sents_list):
    words, pos, rel = set(), set(), set()
    for s in sents_list:
        for line in s:
            words.add(line[0].lower())
            pos.add(line[1])
            rel.add(line[3])

    words = sorted(words)
    pos = sorted(pos)
    rel = sorted(rel)

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

    x2i = {'word' : dict(w2i), 'pos' : dict(p2i), 'rel' : dict(r2i)}
    i2x = {'word' : i2w, 'pos' : i2p, 'rel' : i2r}

    return x2i, i2x


def numericalize_sdp(sents_list, x2i):
    w2i = x2i['word']
    p2i = x2i['pos']
    r2i = x2i['rel']

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


def numericalize_ss(sents_list, x2i):
    w2i = x2i['word']
    p2i = x2i['pos']

    sents_numericalized = []
    for s1, s2 in sents_list:
        new_s1 = np.zeros(s1.shape, dtype=int)
        new_s2 = np.zeros(s2.shape, dtype=int)

        for i in range(len(s1)):
            new_s1[i,0] = w2i.get(s1[i,0].lower(), w2i[UNK_TOKEN])
            new_s1[i,1] = p2i.get(s1[i,1], p2i[UNK_TOKEN])
        for i in range(len(s2)):
            new_s2[i,0] = w2i.get(s2[i,0].lower(), w2i[UNK_TOKEN])
            new_s2[i,1] = p2i.get(s2[i,1], p2i[UNK_TOKEN])

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
