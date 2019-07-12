import random
from random import shuffle
import string
import pickle
import os
import copy

from collections import defaultdict, Counter
import numpy as np
from nltk.parse import CoreNLPParser
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import torch

import utils
from parser import unsort

PAD_TOKEN = '<pad>' # Only place where this seems to come up is in Embeddings, for pad_idx
UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'

CONLLU_MASK = [1, 4, 6, 7]  # [word, pos, head, rel]
CORENLP_URL = 'http://localhost:9000'


def stag_to_sents(f: str):
    sents_list = []

    with open(f, 'r') as stag_file:
        lines = stag_file.readlines()
    if lines[-1] != '\n':
        lines.append('\n') # So split_points works properly
    while lines[0] == '\n':
        lines.pop(0)

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sent_start = 0
    for sent_end in split_points: # Assumes the final line is '\n'
        sents_list.append(lines[sent_start: sent_end])
        sent_start = sent_end + 1 # Skipping the line break

    for i, s in enumerate(sents_list):
        s_split = [line.strip().split(' ') for line in s]
        sents_list[i] = np.array(s_split)
    
    return sents_list


def build_stag_dicts(sents_list):
    s2i = defaultdict(lambda : len(s2i))
    i2s = dict()
    i2s[s2i[UNK_TOKEN]] = UNK_TOKEN

    stags = set()
    for s in sents_list:
        for line in s:
            stags.add(line[2])
    stags = sorted(stags) # Reproducibility
    for stag in stags:
        i2s[s2i[stag]] = stag

    return dict(s2i), i2s


def numericalize_stag(sents_list, x2i):
    w2i = x2i['word']
    p2i = x2i['pos']
    s2i = x2i['stag']

    sents_numericalized = []
    for s in tqdm(sents_list, ascii=True, desc=f'Numericalizing stag data', ncols=80):
        new_s = np.zeros(s.shape, dtype=int)

        for i in range(s.shape[0]):
            new_s[i,0] = w2i.get(s[i,0].lower(), w2i[UNK_TOKEN])
            new_s[i,1] = p2i.get(s[i,1], p2i[UNK_TOKEN])
            new_s[i,2] = s2i.get(s[i,2], s2i[UNK_TOKEN])
            #new_s[i,2] = s2i[s[i,2]] # This would be the more logical way, if not for the fact that there are tags which ONLY occur in the test set

        sents_numericalized.append(new_s)

    return sents_numericalized


def build_stag_dataset(stag_files=[]):
    sents_list = []
    for f in stag_files:
        sents_list.append(stag_to_sents(f))

    # The 'standard' split for CCGBank
    train_list = [s for f in sents_list[2:22] for s in f] # 02-21 for train
    dev_list = sents_list[0]
    test_list = sents_list[23]
    
    #s2i, i2s = build_stag_dicts(train_list)
    l = []
    l.extend(train_list)
    l.extend(dev_list)
    l.extend(test_list)
    s2i, i2s = build_stag_dicts(l)

    raw_data = {'train': train_list,
            'dev': dev_list,
            'test': test_list}

    return raw_data, s2i, i2s
    

def build_ptb_dataset(conllu_files=[], filter_sents=False):
    if len(conllu_files) != 24:
        print(f'Missing a conllu file? {len(sents_list)} files provided.')
        raise Exception

    sents_list = []
    for f in conllu_files:
        sents_list.append(conllu_to_sents(f))

    for i, f in enumerate(sents_list):
        sents_list[i] = [s[:, CONLLU_MASK] for s in f]

    # "Standard" train/dev/split for PTB
    train_list = [s for f in sents_list[2:22] for s in f] # 02-21 for train
    dev_list = sents_list[22]
    test_list = sents_list[23]

    word_counts = get_word_counts(train_list)

    if filter_sents:
        filter_sentences(train_list, word_counts=word_counts)
        filter_sentences(dev_list)
        filter_sentences(test_list)

    x2i_ptb, i2x_ptb = build_dicts(train_list)

    raw_data = {'train': train_list, 'dev': dev_list, 'test': test_list}

    return raw_data, x2i_ptb, i2x_ptb, word_counts


def build_sdp_dataset(conllu_files: list, x2i=None, filter_sents=False):
    '''
        For building a dataset from conllu files in general, once the x2i dict
        has been constructed by build_ptb_dataset.
    '''
    data = {}
    word_counts = {}

    for f in conllu_files:
        name = os.path.splitext(f)[0].split('/')[-1].lower()
        data[name] = conllu_to_sents(f)

    for name, sents in data.items():
        sents_masked = [s[:, CONLLU_MASK] for s in sents]
        if filter_sents:
            filter_sentences(sents_masked)

        word_counts[name] = get_word_counts(sents_masked)

        data[name] = numericalize_sdp(sents_masked, x2i)

    return data, word_counts


def build_ss_dataset(raw_sent_pairs, gs='', x2i=None, filter_sents=False):
    word_counts = None
    if filter_sents:
        flattened_raw = []
        for s1, s2 in raw_sent_pairs:
            flattened_raw.append(s1)
            flattened_raw.append(s2)
        filter_sentences(flattened_raw) # in-place
        word_counts = get_word_counts(flattened_raw)

    numericalized_pairs = numericalize_ss(raw_sent_pairs, x2i)

    raw_targets = txt_to_sem_scores(gs) if gs else None
    
    sent_pairs = []
    targets = []
    if raw_targets != None:
        for s, t in zip(numericalized_pairs, raw_targets):
            sent_pairs.append(s)
            targets.append(t)
        if len(targets) != len(sent_pairs):
            print('Mismatch between targets ({len(targets)}) and sents ({len(sent_pairs)})')
            raise Exception
    else:
        sent_pairs = numericalized_pairs

    return {'sent_pairs': sent_pairs, 'targets': targets, 'word_counts': word_counts}


def txt_to_sem_scores(txt: str) -> list:
    with open(txt, 'r') as f:
        lines = f.readlines()
        sem_scores = [float(l.strip()) for l in lines if l != '\n']

    return sem_scores


def stag_data_loader(data, batch_size=None, shuffle_idx=False):
    idx = list(range(len(data)))

    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle
        for chunk in idx_chunks(idx, batch_size):
            batch = [data[i] for i in chunk]
            yield prepare_batch_stag(batch)


def sdp_data_loader(data, batch_size=None, shuffle_idx=False):
    idx = list(range(len(data)))

    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle
        
        for chunk in idx_chunks(idx, batch_size):
            batch = [data[i] for i in chunk]
            yield prepare_batch_sdp(batch)


def idx_loader(num_data=None, batch_size=None):
    idx = list(range(num_data))
    while True:
        shuffle(idx)
        for chunk in idx_chunks(idx, batch_size):
            yield chunk


def idx_chunks(idx, chunk_size):
    for i in range(0, len(idx), chunk_size):
        yield idx[i:i+chunk_size]


def prepare_batch_sdp(batch):
    batch_size = len(batch)
    sent_lens = torch.LongTensor([s.shape[0] for s in batch]) # Keep in mind, these lengths include ROOT token in each sentence
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    pos = torch.zeros((batch_size, l_longest)).long()
    arc_targets = torch.LongTensor(batch_size, l_longest).fill_(-1)
    rel_targets = torch.LongTensor(batch_size, l_longest).fill_(-1)

    dt = np.dtype(int)
    for i, s in enumerate(batch):
        resized_input = np.zeros((l_longest, 2), dtype=dt)
        resized_target = np.zeros((l_longest, 2), dtype=dt) - 1
        resized_input[:s.shape[0]] = s[:,:2]
        resized_target[:s.shape[0]] = s[:,2:]
        words[i] = torch.LongTensor(resized_input[:,0])
        pos[i] = torch.LongTensor(resized_input[:,1])
        arc_targets[i] = torch.LongTensor(resized_target[:,0])
        rel_targets[i] = torch.LongTensor(resized_target[:,1])

    return {'words': words, 
            'pos' : pos, 
            'sent_lens' : sent_lens, 
            'arc_targets' : arc_targets, 
            'rel_targets' : rel_targets}


def prepare_batch_stag(batch):
    batch_size = len(batch)
    sent_lens = torch.LongTensor([len(s) for s in batch])
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    pos = torch.zeros((batch_size, l_longest)).long()
    stag_targets = torch.LongTensor(batch_size, l_longest).fill_(-1)

    dt = np.dtype(int)
    for i, s in enumerate(batch):
        s_len = len(s)
        resized_words = np.zeros(l_longest, dtype=dt)
        resized_pos = np.zeros(l_longest, dtype=dt)
        resized_stags = np.zeros(l_longest, dtype=dt) - 1
        resized_words[:s_len] = s[:,0]
        resized_pos[:s_len] = s[:,1]
        resized_stags[:s_len] = s[:,2]
        words[i] = torch.LongTensor(resized_words)
        pos[i] = torch.LongTensor(resized_pos)
        stag_targets[i] = torch.LongTensor(resized_stags)

    return {'words': words,
            'pos' : pos,
            'stag_targets' : stag_targets,
            'sent_lens' : sent_lens}


def prepare_batch_ss(batch):
    batch_size = len(batch)
    sent_lens = torch.LongTensor([len(s) for s in batch])
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    pos = torch.zeros((batch_size, l_longest)).long()

    dt = np.dtype(int)
    for i, s in enumerate(batch):
        resized_s = np.zeros((l_longest, s.shape[1]), dtype=dt)
        resized_s[:len(s)] = s
        words[i] = torch.LongTensor(resized_s[:,0])
        pos[i] = torch.LongTensor(resized_s[:,1])

    return words, pos, sent_lens


def conllu_to_sents(f: str):
    sents_list = []

    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()
    if lines[-1] != '\n':
        lines.append('\n') # So split_points works properly
    while lines[0] == '\n':
        lines.pop(0)

    split_points = [idx for idx, line in enumerate(lines) if line == '\n']

    sent_start = 0
    for sent_end in split_points: # Assumes the final line is '\n'
        sents_list.append(lines[sent_start: sent_end])
        sent_start = sent_end + 1 # Skipping the line break

    for i, s in enumerate(sents_list):
        s_split = [line.rstrip().split('\t') for line in s]
        sents_list[i] = np.array(s_split)

    return sents_list


def paraphrase_to_sents(f: str):
    '''
        inputs:
            f - name of sentences/paraphrases dataset txt file

        outputs:
            sent_pairs - a list of pairs (tuples) of sentences and their
                         paraphrases
    '''

    # TODO Some kind of try/catch here if server connection fails?
    tagger = CoreNLPParser(url=f'{CORENLP_URL}', tagtype='pos')

    with open(f, 'r') as para_file:
        lines = para_file.readlines()

    sent_pairs = []
    for line in tqdm(lines, ascii=True, desc=f'Paraphrase file to sentences', ncols=80):
        try:
            sents = line.split('\t')
            s1 = sents[0].strip().split(' ')
            s2 = sents[1].strip().split(' ')
            s1 = np.array(tagger.tag(s1))
            s2 = np.array(tagger.tag(s2))
            sent_pairs.append( (s1,s2) )
        except Exception:
            print(f'Problem pair is:\n{s1}\n{s2}')
            continue

    return sent_pairs


def build_dicts(sents_list):
    word, pos, rel = set(), set(), set()
    for s in sents_list:
        for line in s:
            word.add(line[0].lower())
            pos.add(line[1])
            rel.add(line[3])

    word = sorted(word)
    pos = sorted(pos)
    rel = sorted(rel)

    w2i = defaultdict(lambda : len(w2i))
    p2i = defaultdict(lambda : len(p2i))
    r2i = defaultdict(lambda : len(r2i))
    i2w, i2p, i2r = dict(), dict(), dict()

    i2w[w2i[PAD_TOKEN]] = PAD_TOKEN
    i2p[p2i[PAD_TOKEN]] = PAD_TOKEN

    i2w[w2i[UNK_TOKEN]] = UNK_TOKEN
    i2p[p2i[UNK_TOKEN]] = UNK_TOKEN

    i2w[w2i[ROOT_TOKEN]] = ROOT_TOKEN
    i2p[p2i[ROOT_TOKEN]] = ROOT_TOKEN

    for w in word:
        i2w[w2i[w]] = w
    for p in pos:
        i2p[p2i[p]] = p
    for r in rel:
        i2r[r2i[r]] = r

    x2i = {'word' : dict(w2i), 'pos' : dict(p2i), 'rel' : dict(r2i)}
    i2x = {'word' : i2w, 'pos' : i2p, 'rel' : i2r}

    return x2i, i2x


def build_pretrained_w2v(word_v_file=None):
    w2v = {}

    print(f'Building w2v from file: {word_v_file}')

    with open(word_v_file, 'r', errors='ignore') as f:
        lines = f.readlines()

        if len(lines[0].split()) == 2:
            lines.pop(0)

        for i, line in tqdm(enumerate(lines), ascii=True, desc=f'w2v Progress', ncols=80):
            line = line.split()
            word = line[0].lower()
            try:
                word_vector = [float(value) for value in line[1:]]
                w2v[word] = word_vector
            except Exception:
                print(f'Word is: {word}, line is {i}')

    return w2v


def build_embedding_data(w2v):
    w2i = defaultdict(lambda : len(w2i))
    i2w = {}
    word_v_list = []

    e_size = len(w2v['the']) # Should be the same length as any other word vector
    
    i2w[w2i[PAD_TOKEN]] = PAD_TOKEN
    word_v_list.append([0] * e_size)
    i2w[w2i[UNK_TOKEN]] = UNK_TOKEN
    word_v_list.append([0] * e_size)
    i2w[w2i[ROOT_TOKEN]] = ROOT_TOKEN # May or may not be important
    word_v_list.append([0] * e_size)

    for word, embedding in tqdm(w2v.items(), ascii=True, desc=f'Building embedding data', ncols=80):
        i2w[w2i[word]] = word
        word_v_list.append(embedding)

    word_e = np.array(word_v_list)

    return {'w2i': dict(w2i),
            'i2w': i2w,
            'word_e': word_e}


def numericalize_sdp(sents_list, x2i):
    w2i = x2i['word']
    p2i = x2i['pos']
    r2i = x2i['rel']

    unk_count = 0 # XXX
    sents_numericalized = []
    for s in tqdm(sents_list, ascii=True, desc=f'Numericalizing SDP data', ncols=80):
        new_shape = (s.shape[0] + 1, s.shape[1])

        new_s = np.zeros(new_shape, dtype=int) # Making room for ROOT_TOKEN
        new_s[0,:] = w2i[ROOT_TOKEN], p2i[ROOT_TOKEN], -1, -1 # -1s here become crucial for attachment scoring

        for i in range(s.shape[0]):
            word_i = w2i.get(s[i,0].lower(), w2i[UNK_TOKEN])
            if word_i == 1:
                unk_count += 1
            #new_s[i+1,0] = w2i.get(s[i,0].lower(), w2i[UNK_TOKEN])
            new_s[i+1,0] = word_i
            new_s[i+1,1] = p2i.get(s[i,1], p2i[UNK_TOKEN])
            new_s[i+1,2] = int(s[i,2]) # Head idx
            new_s[i+1,3] = r2i[s[i,3]]

        sents_numericalized.append(new_s)

    print(f'Unk count is {unk_count}') #XXX
    return sents_numericalized


def numericalize_ss(sents_list, x2i):
    w2i = x2i['word']
    p2i = x2i['pos']

    sents_numericalized = []
    for s1, s2 in tqdm(sents_list, ascii=True, desc=f'Numericalizing SS data', ncols=80):
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


def decode_sdp_sents(sents=[], i2x=None) -> list:
    i2w = i2x['word']
    i2p = i2x['pos']
    i2r = i2x['rel']

    decoded_sents = []
    for sent in sents:
        words = []
        pos = []
        heads = []
        rels = []

        for i in range(sent.shape[0]):
            words.append(i2w[sent[i,0]])
            pos.append(i2p[sent[i,1]])
            heads.append(sent[i,2])
            rel = i2r[sent[i,3]] if sent[i,3] != -1 else ROOT_TOKEN
            rels.append(rel)

        decoded_sents.append([words, pos, heads, rels])

    return decoded_sents


#According to Weiting, effective in regularization
def scramble_words(batch, scramble_prob=0.):
    if scramble_prob > 0:
        n = np.random.binomial(1, scramble_prob, len(batch))
        for i, outcome in enumerate(n):
            if outcome == 1:
                copy = batch[i].copy()
                np.random.shuffle(copy)
                batch[i] = copy


def megabatch_breakdown(megabatch, minibatch_size=None, parser=None, args=None, data=None):
    '''
        inputs:
            megabatch - an unprepared megabatch (M many batches) of sentences
            batch_size - size of a minibatch

        outputs:
            s1 - list of orig. sentence instances
            mb_s2 - list of paraphrase instances
            mb_n1 - list of neg sample instances
    '''
    device = data['device']

    mb_s1 = []
    mb_s2 = []
    
    for s1, s2 in megabatch:
        mb_s1.append(s1) # Does this allocate new memory?
        mb_s2.append(s2)

    if args.scramble > 0:
        scramble_words(mb_s1, scramble_prob=args.scramble)
        scramble_words(mb_s2, scramble_prob=args.scramble)

    minibatches_s1 = [mb_s1[i:i+minibatch_size] for i in range(0, len(mb_s1), minibatch_size)]
    minibatches_s2 = [mb_s2[i:i+minibatch_size] for i in range(0, len(mb_s2), minibatch_size)]

    check = 0
    for mini in minibatches_s1:
        check += len(mini)
    if check != len(megabatch):
        print(len(minibatches_s1) * minibatch_size)
        print(len(megabatch))
        raise Exception

    mb_s1_reps = [] # (megabatch_size, )
    mb_s2_reps = [] 
    pi = parser.pos_in
    for b1, b2 in zip(minibatches_s1, minibatches_s2):
        w1, p1, sl1 = prepare_batch_ss(b1)
        sl1 = sl1.to(device)
        packed_b1, idx_b1, _, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device) if pi else None)
        b1_reps = unsort(parser.SemanticRNN(packed_b1), idx_b1)
        b1_reps_avg = utils.average_hiddens(b1_reps, sl1, sum_f_b=args.sum_f_b)
        mb_s1_reps.append(b1_reps_avg)
        if args.two_negs:
            w2, p2, sl2 = prepare_batch_ss(b2)
            sl2 = sl2.to(device)
            packed_b2, idx_b2, _, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device) if pi else None)
            packed_b2, idx_b2, _, _ = parser.Embeddings(w2.to(device), sl2)
            b2_reps = unsort(parser.SemanticRNN(packed_b2), idx_b2)
            b2_reps_avg = utils.average_hiddens(b2_reps, sl2, sum_f_b=args.sum_f_b)
            mb_s2_reps.append(b2_reps_avg)

    # Stack all reps into torch tensors
    mb_s1_reps = torch.cat(mb_s1_reps)
    mb_s2_reps = torch.cat(mb_s2_reps) if args.two_negs else None

    # Get negative samples with respect to mb_s1
    mb_n1, mb_n2 = get_negative_samps(megabatch, mb_s1_reps, mb_s2_reps)

    return mb_s1, mb_s2, mb_n1, mb_n2


def get_negative_samps(megabatch, mb_s1_reps, mb_s2_reps):
    '''
        inputs:
            megabatch - a megabatch (list) of sentences
            megabatch_of_reps - a tensor of sentence representations

        outputs:
            neg_samps - a list matching length of input megabatch consisting
                        of sentences
    '''

    two_negs = mb_s2_reps is not None

    reps = []
    sents = []
    for i, (s1, s2) in enumerate(megabatch):
        reps.append(mb_s1_reps[i].cpu().numpy())
        sents.append(s1)
        if two_negs:
            reps.append(mb_s2_reps[i].cpu().numpy())
            sents.append(s2)

    dists = pdist(reps, 'cosine') # cosine distance, as (1 - normalized inner product)
    dists = squareform(dists) # Symmetric 2-D matrix of pairwise distances

    # Don't risk pairing a sentence with itself
    if two_negs:
        i1 = np.arange(0, dists.shape[0], 2)
        i2 = np.arange(1, dists.shape[0], 2)
        if len(i1) != len(i2):
            raise Exception
        dists[i1, i2] = 3
        dists[i2, i1] = 3
    np.fill_diagonal(dists, 3)
    
    # For each sentence, get index of sentence 'closest' to it
    neg_idxs = np.argmin(dists, axis=1)

    mb_n1 = []
    mb_n2 = [] if two_negs else None
    for idx in range(len(megabatch)):
        if two_negs:
            n1 = sents[neg_idxs[2*idx]]
            n2 = sents[neg_idxs[(2*idx)+1]]
            mb_n1.append(n1)
            mb_n2.append(n2)
        else:
            n1 = sents[neg_idxs[idx]]
            mb_n1.append(n1)

    return mb_n1, mb_n2


# From https://github.com/EelcovdW/Biaffine-Parser/blob/master/data_utils.py
def filter_sentences(sentences, word_counts=None):
    # In-place filtering of the sentences as numpy arrays of strings
    if word_counts is not None:
        one_words = set([w for w, c in word_counts.items() if c == 1])
    for sentence in tqdm(sentences, ascii=True, desc=f'Progress in filtering.', ncols=80):
        for unit in sentence:
            word = unit[0]
            if is_url(word):
                unit[0] = '<url>'
            elif is_long_punctuation(word):
                unit[0] = '<punct>'
            elif has_digits(word):
                unit[0] = '<num>'
            elif word_counts is not None and word.lower() in one_words:
                unit[0] = UNK_TOKEN


def get_word_counts(sentences):
    words = [unit[0].lower() for sentence in sentences for unit in sentence]
    return Counter(words)


def is_url(word):
    return bool(set('./:').issubset(word))


def is_long_punctuation(word):
    return bool(len(word) > 2 and set(string.punctuation).issuperset(word))


def has_digits(word):
    return bool(set(string.digits).intersection(word))
