import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict


UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'


'''
   Things that still need doing:
   1. Shuffling the data
   2. A way of splitting to train/dev/test
   3. Usage of num2x_maps dict anywhere
   4. Properly filtering words


'''


def get_dataset(conllu_file):
        sents_list = conllu_to_sents(conllu_file)

        dicts, _ = build_dicts(sents_list)

        data_list = numericalize(sents_list, dicts)

        word_vsize = len(dicts['word'])
        pos_vsize = len(dicts['pos'])
        rel_vsize = len(dicts['rel'])

        return data_list, word_vsize, pos_vsize, rel_vsize


#TODO Possible edit required: EelcovdW's implementation
# uses chunks of shuffled INDICES rather than chunks of the 
# data list itself; this may be a more convenient/efficient
# implementation once I get to the point of SHUFFLING the data
def custom_data_loader(data_list, batch_size):

    while True:
        for chunk in chunk_generator(data_list, batch_size):
            yield chunk_to_batch(chunk)


def chunk_to_batch(chunk):
    '''
    Transform a chunk (list) of sentences
    into a batch for training the parser
    in the form of five LongTensors representing
    the sentences as words, pos, arcs, deprels

    requires PADDING
    '''

    batch_size = len(chunk)
    chunk_sorted = sorted(chunk, key = lambda s: np.shape(s)[0], reverse=True)
    sent_lens = [np.shape(s)[0] for s in chunk_sorted]
    length_longest = sent_lens[0]

    #By using zeros tensors, implicitly filling with padding
    words = torch.zeros((batch_size, length_longest), dtype=torch.long)
    pos = torch.zeros((batch_size, length_longest), dtype=torch.long)
    heads = torch.zeros((batch_size, length_longest), dtype=torch.long)
    rels = torch.zeros((batch_size, length_longest), dtype=torch.long)
    for i, s in enumerate(chunk_sorted):
        for j, _ in enumerate(s):
            '''
            Casting as ints because for some stupid reason
            you cannot set a value in torch long tensor using 
            numpy's 64 bit ints
            '''
            words[i,j] = int(s[j,0])
            pos[i,j] = int(s[j,1])
            heads[i,j] =  int(s[j,2])
            rels[i,j] =  int(s[j,3])


    return words, pos, sent_lens, heads, rels


def chunk_generator(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]


'''NOTE:
Probably we will call this once for a 
separate conllu file for the train, dev, and test sets
'''
def conllu_to_sents(f: str):
    '''
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)

    '''

    mask = [1, 4, 6, 7] # [word, pos, head, rel]

    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sents_list = []
    sent_start = 0
    for sent_end in split_points:
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
    num2rel[rel2num[PAD_TOKEN]] = PAD_TOKEN

    num2word[word2num[UNK_TOKEN]] = UNK_TOKEN

    num2rel[rel2num[ROOT_TOKEN]] = ROOT_TOKEN

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

    sents_num = []

    for s in sents_list:
        curr = np.zeros(np.shape(s), dtype=int)

        for i in range(np.shape(s)[0]):
            curr[i,0] = word2num[s[i, 0].lower()]
            curr[i,1] = pos2num[s[i, 1]]
            curr[i,2] = int(s[i, 2]) #head
            curr[i,3] = rel2num[s[i, 3]]

        sents_num.append(curr)

    return sents_num

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

