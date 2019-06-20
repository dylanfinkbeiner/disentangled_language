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

UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'

CONLLU_MASK = [1, 4, 6, 7]  # [word, pos, head, rel]
CORENLP_URL = 'http://localhost:9000'


#def compare(conllus=[], stags=[]):
#    for c, s in zip(conllus, stags):
#
#        c_sents = conllu_to_sents(c)
#        s_sents = stag_to_sents(s)
#
#        while(len(c_sents) != 0):
#            s1 = c_sents[0]
#            s2 = s_sents[0]
#
#            if len(s1) != len(s2):
#                print(f'Problem in files {c} and {s}\nProblem sentences:\n{s1}\n{s2}')
#                breakpoint()
#                #break
#
#            c_sents.pop(0)
#            s_sents.pop(0)

#        for i in range(len(s_sents)):
#            s1 = c_sents[i]
#            s2 = s_sents[i]
#
#            for j in range(len(s1)):
#                if s1[j][1] != s2[j][0]:
#                    print(f'Problem in files {c} and {s}\nProblem sentences:\n{s1}\n{s2}')
#                    breakpoint()
#                    break


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
    stag = set()
    for s in sents_list:
        for line in s:
            stag.add(line[1])

    stag = sorted(stag)

    s2i = defaultdict(lambda : len(s2i))
    i2s = dict()

    #Crucial that PAD_TOKEN map to 0 so that chunk_to_batch() definition correct
    i2s[s2i[PAD_TOKEN]] = PAD_TOKEN
    s2i[s2i[UNK_TOKEN]] = UNK_TOKEN

    for s in stag:
        i2s[s2i[s]] = s

    return dict(s2i), i2s


def numericalize_stag(sents_list, x2i):
    w2i = x2i['word']
    s2i = x2i['stag']

    sents_numericalized = []
    for s in tqdm(sents_list, ascii=True, desc=f'Numericalizing stag data', ncols=80):
        new_s = np.zeros(s.shape, dtype=int) # Making room for ROOT_TOKEN

        for i in range(s.shape[0]):
            new_s[i,0] = w2i.get(s[i,0].lower(), w2i[UNK_TOKEN])
            new_s[i,1] = s2i.get(s[i,1], s2i[UNK_TOKEN])

        sents_numericalized.append(new_s)

    return sents_numericalized


def build_ptb_stags(stag_files=[]):
    sents_list = []
    for f in stag_files:
        sents_list.append(stag_to_sents(f))

    train_list = [s for f in sents_list[2:22] for s in f] #02-21 for train
    dev_list = sents_list[22]
    test_list = sents_list[23]

    s2i, i2s = build_stag_dicts(train_list)

    raw_data = {'train': train_list,
            'dev': dev_list,
            'test': test_list}

    return raw_data, s2i, i2s
    

def build_ptb_dataset(conllu_files=[], filter_sents=False):
    '''
        inputs:
            conllu_files - a list of sorted strings, filenames of dependencies

        output:
            conllu_files - a list
    '''
    sents_list = []

    if not conllu_files:
        print(f'Empty list of filenames passed.')
        raise Exception

    for f in conllu_files:
        sents_list.append(conllu_to_sents(f))

    if len(sents_list) != 24:
        print(f'Missing a conllu file? {len(sents_list)} files provided.')
        raise Exception

    #XXX
    for i, c in enumerate(sents_list):
        print(f'{i} contains {len(c)} sentences')
    exit()
    #XXX

    for i, f in enumerate(sents_list):
        sents_list[i] = [s[:, CONLLU_MASK] for s in f]


    # "Standard" train/dev/split for PTB
    train_list = [s for f in sents_list[2:22] for s in f] #02-21 for train
    dev_list = sents_list[22] #22
    test_list = sents_list[23]

    word_counts = get_word_counts(train_list)

    if filter_sents:
        filter_sentences(train_list, word_counts=word_counts)
        filter_sentences(dev_list)
        filter_sentences(test_list)

    x2i_ptb, i2x_ptb = build_dicts(train_list)

    raw_data = {'train': train_list,
            'dev': dev_list,
            'test': test_list}

    return raw_data, x2i_ptb, i2x_ptb, word_counts


def build_sdp_dataset(conllu_files: list, x2i=None, filter_sents=False):
    '''
        For building a dataset from conllu files in general, once the x2i dict
        has been constructed by build_ptb_dataset.
    '''
    data = {}

    for f in conllu_files:
        name = os.path.splitext(f)[0].split('/')[-1].lower()
        data[name] = conllu_to_sents(f)

    for name, sents in data.items():
        sents_masked = [s[:, CONLLU_MASK] for s in sents]
        if filter_sents:
            filter_sentences(sents_masked)
        data[name] = numericalize_sdp(sents_masked, x2i)

    return data


def build_ss_dataset(raw_sent_pairs, gs='', x2i=None, filter_sents=False):
    word_counts = None
    if filter_sents:
        flattened_raw = []
        for s1, s2 in raw_sent_pairs:
            flattened_raw.append(s1)
            flattened_raw.append(s2)
        filter_sentences(flattened_raw)
        word_counts = get_word_counts(flattened_raw)

    # Raw sent pairs got modified within filter and count
    numericalized_pairs = numericalize_ss(raw_sent_pairs, x2i)

    raw_targets = txt_to_sem_scores(gs) if gs else None
    
    sent_pairs = []
    targets = []
    if raw_targets != None:
        for s, t in zip(numericalized_pairs, raw_targets):
            if t != -1.0:
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
        sem_scores = [float(l.strip()) if l != '\n' else -1.0 for l in lines]

    return sem_scores


def build_cutoff_dicts(sents_sorted: list) -> dict:
    '''
        inputs:
            sents_sorted - list of np arrays (conllu-formatted sentences)

        returns:
            a dictionary i2c, keys are indices in sorted data, values are lists with 2 elements,
            the first index in sents_sorted of a sentence of that length and the 
            (non-inclusive) final index
    '''
    i2c = dict()
    l2c = defaultdict(list)
    l2n = defaultdict(int)

    l_prev = sents_sorted[0].shape[0]
    l_max = sents_sorted[-1].shape[0]
    l2c[l_prev].append(0)
    l2n[l_prev] += 1
    for i, s in enumerate(sents_sorted[1:], start=1):
        l = s.shape[0]
        l2n[l] += 1
        if l > l_prev:
            l2c[l_prev].append(i)
            l2c[l].append(i)
        l_prev = l
    l2c[l_max].append(len(sents_sorted))

    for c in l2c.values():
        for i in range(c[0], c[1]):
            i2c[i] = c

    if len(i2c) != len(sents_sorted):
        print(f'i2c {len(i2c)} != sents_sorted {len(sents_sorted)}')
        raise Exception


    return {'l2c': dict(l2c), 'i2c': i2c, 'l2n': dict(l2n)}


def get_paired_idx(idx: list, cutoffs: dict):
    '''
        produces a list of indices, paired to an index in idx, of a
        sentence of equal length
    '''
    paired_idx = []
    for i in idx:
        c = cutoffs[i]
        paired_i = random.randrange(c[0], c[1])
        is_unique_length = (c[0] == c[1] - 1)
        while (paired_i == i) and not is_unique_length:
            paired_i = random.randrange(c[0], c[1])
        paired_idx.append(paired_i)

    return paired_idx


def get_syntactic_scores(s1_batch, s2_batch, device=None):
    ''' Not a great name for the function: must change this later... '''
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results = utils.attachment_scoring(
            arc_preds=s1_batch['arc_targets'].to(device), 
            rel_preds=s1_batch['rel_targets'].to(device), 
            arc_targets=s2_batch['arc_targets'].to(device), 
            rel_targets=s2_batch['rel_targets'].to(device), 
            sent_lens=s1_batch['sent_lens'].to(device), 
            include_root=False,
            keep_dim=True)

    return results


#def length_to_results(data_sorted, l2c=None, device=None, chunk_size=100) -> dict:
#    ''' 'results', here, refers to the output of my get_syntactic_scores function '''
#
#    l2r = {}
#
#    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building l2r', ncols=80):
#        idxs = list(range(c[0], c[1]))
#        n = len(idxs)
#
#        UAS_chunks = []
#        LAS_chunks = []
#        for i, idx_i in enumerate(idxs[:-2]):
#            s1_batch = []
#            s2_batch = []
#            for idx_j in idxs[i+1:]:
#                s1_batch.append(data_sorted[idx_i])
#                s2_batch.append(data_sorted[idx_j])
#        
#            results = get_syntactic_scores(
#                    prepare_batch_sdp(s1_batch),
#                    prepare_batch_sdp(s2_batch),
#                    device=device)
#
#            UAS_chunks.append(results['UAS'].flatten()) # (chunk_size) shaped tensor
#            LAS_chunks.append(results['LAS'].flatten())
#
#        # Stack up results from batched attachment scoring
#        UAS_l = torch.cat(UAS_chunks, dim=0)
#        LAS_l = torch.cat(LAS_chunks, dim=0)
#        expected_len = (n * (n-1)) / 2
#        if UAS_l.shape[0] != expected_len:
#            print(f'Expected len: {expected_len}, UAS_l len: {UAS_l.shape[0]}')
#            raise Exception
#        elif LAS_l.shape[0] != expected_len:
#            print(f'Expected len: {expected_len}, LAS_l len: {LAS_l.shape[0]}')
#            raise Exception
#
#        l2r[l] = {'UAS': UAS_l, 'LAS': LAS_l}
#
#    return l2r


def build_l2p(sents_sorted, l2c=None):
    l2p = {}

    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building l2p', ncols=80):
        idxs = list(range(c[0], c[1]))
        pairs = []

        for i, idx_i in enumerate(idxs[:-2]):
            s1_batch = []
            s2_batch = []
            for idx_j in idxs[i+1:]:
                s1_batch.append(sents_sorted[idx_i])
                s2_batch.append(sents_sorted[idx_j])

            results = get_syntactic_scores(
                    prepare_batch_sdp(s1_batch),
                    prepare_batch_sdp(s2_batch))

            uas_batch = results['UAS'].flatten().tolist() # (chunk_size) shaped tensor
            las_batch = results['LAS'].flatten().tolist()

            for idx_j, uas, las in zip(idxs[i+1:], uas_batch, las_batch):
                pairs.append( (idx_i, idx_j, uas, las) )

        l2p[l] = pairs
        print(f'Pairs size is {len(pairs)}')

    return l2p


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
    '''
        inputs:
            data - the full Python list of pairs of numericalized sentences (np arrays)
            batch_size - batch size

        yields:
            chunk - list of indices representing a minibatch
    '''
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
    sent_lens = torch.LongTensor([s.shape[0] for s in batch])
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    stags = torch.LongTensor(batch_size, l_longest).fill_(-1)

    dt = np.dtype(int)
    for i, s in enumerate(batch):
        resized_input = np.zeros(l_longest, dtype=dt)
        resized_target = np.zeros(l_longest, dtype=dt) - 1
        resized_input[:s.shape[0]] = s[:,0]
        resized_target[:s.shape[0]] = s[:,1]
        words[i] = torch.LongTensor(resized_input)
        stags[i] = torch.LongTensor(resized_target)

    return {'words': words, 
            'stags' : stags, 
            'sent_lens' : sent_lens}


def prepare_batch_ss(batch):
    batch_size = len(batch)

    sent_lens = torch.LongTensor([s.shape[0] for s in batch])
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    pos = torch.zeros((batch_size, l_longest)).long()

    for i, s in enumerate(batch):
        dt = np.dtype(int)
        resized_np = np.zeros((l_longest, s.shape[1]), dtype=dt)
        resized_np[:s.shape[0]] = s
        words[i] = torch.LongTensor(resized_np[:,0])
        pos[i] = torch.LongTensor(resized_np[:,1])

    return words, pos, sent_lens


def conllu_to_sents(f: str):
    '''
    outputs:
        sents_list - a list of np arrays, each with shape (#words-in-sentence, 4)
    '''
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


def build_dicts(sents_list, is_sdp=True):
    if not is_sdp:
        paired_sents_list = sents_list
        sents_list = []
        for s1, s2 in paired_sents_list:
            sents_list.append(s1)
            sents_list.append(s2)

    word, pos, rel = set(), set(), set()
    for s in sents_list:
        for line in s:
            word.add(line[0].lower())
            pos.add(line[1])
            if is_sdp:
                rel.add(line[3])

    word = sorted(word)
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

    sents_numericalized = []
    for s in tqdm(sents_list, ascii=True, desc=f'Numericalizing SDP data', ncols=80):
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
def scramble_words(batch, scramble_prob=0.3):
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
            mb_para2 - list of paraphrase instances
            mb_neg1 - list of neg sample instances
    '''
    device = data['device']

    mb_para1 = []
    mb_para2 = []
    
    for para1, para2 in megabatch:
        mb_para1.append(para1) # Does this allocate new memory?
        mb_para2.append(para2)

    if args.scramble > 0:
        scramble_words(mb_para1, scramble_prob=args.scramble)
        scramble_words(mb_para2, scramble_prob=args.scramble)

    minibatches_para1 = [mb_para1[i:i+minibatch_size] for i in range(0, len(mb_para1), minibatch_size)]
    minibatches_para2 = [mb_para2[i:i+minibatch_size] for i in range(0, len(mb_para2), minibatch_size)]

    if len(minibatches_para1) * minibatch_size != len(megabatch):
        raise Exception

    mb_para1_reps = [] # (megabatch_size, )
    mb_para2_reps = [] 
    for b1, b2 in zip(minibatches_para1, minibatches_para2):
        #w1, p1, sl1 = prepare_batch_ss(b1)
        w1, _, sl1 = prepare_batch_ss(b1)
        sl1 = sl1.to(device)
        #packed_b1, idx_b1, _ = parser.Embeddings(w1.to(device), sl1, pos=p1.to(device))
        packed_b1, idx_b1, _ = parser.Embeddings(w1.to(device), sl1)
        b1_reps = unsort(parser.SemanticRNN(packed_b1), idx_b1)
        b1_reps_avg = utils.average_hiddens(b1_reps, sl1, sum_f_b=args.sum_f_b)
        mb_para1_reps.append(b1_reps_avg)
        if args.two_negs:
            #w2, p2, sl2 = prepare_batch_ss(b2)
            w2, _, sl2 = prepare_batch_ss(b2)
            sl2 = sl2.to(device)
            #packed_b2, idx_b2, _ = parser.Embeddings(w2.to(device), sl2, pos=p2.to(device))
            packed_b2, idx_b2, _ = parser.Embeddings(w2.to(device), sl2)
            b2_reps = unsort(parser.SemanticRNN(packed_b2), idx_b2)
            b2_reps_avg = utils.average_hiddens(b2_reps, sl2, sum_f_b=args.sum_f_b)
            mb_para2_reps.append(b2_reps_avg)

    # Stack all reps into torch tensors
    mb_para1_reps = torch.cat(mb_para1_reps)
    mb_para2_reps = torch.cat(mb_para2_reps) if args.two_negs else None

    # Get negative samples with respect to mb_para1
    mb_neg1, mb_neg2 = get_negative_samps(megabatch, mb_para1_reps, mb_para2_reps)

    return mb_para1, mb_para2, mb_neg1, mb_neg2


def get_negative_samps(megabatch, mb_para1_reps, mb_para2_reps):
    '''
        inputs:
            megabatch - a megabatch (list) of sentences
            megabatch_of_reps - a tensor of sentence representations

        outputs:
            neg_samps - a list matching length of input megabatch consisting
                        of sentences
    '''

    two_negs = mb_para2_reps is not None

    reps = []
    sents = []
    for i, (para1, para2) in enumerate(megabatch):
        reps.append(mb_para1_reps[i].cpu().numpy())
        sents.append(para1)
        if two_negs:
            reps.append(mb_para2_reps[i].cpu().numpy())
            sents.append(para2)

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

    mb_neg1 = []
    mb_neg2 = [] if two_negs else None
    for idx in range(len(megabatch)):
        if two_negs:
            neg1 = sents[neg_idxs[2*idx]]
            neg2 = sents[neg_idxs[(2*idx)+1]]
            mb_neg1.append(neg1)
            mb_neg2.append(neg2)
        else:
            neg1 = sents[neg_idxs[idx]]
            mb_neg1.append(neg1)

    return mb_neg1, mb_neg2


# From https://github.com/EelcovdW/Biaffine-Parser/blob/master/data_utils.py
def filter_sentences(sentences, word_counts=None):
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

