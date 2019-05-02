import random
from random import shuffle
import string
import pickle
import os

from collections import defaultdict, Counter
import numpy as np
from nltk.parse import CoreNLPParser
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import torch

import utils

UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'

CONLLU_MASK = [1, 4, 6, 7]  # [word, pos, head, rel]
CORENLP_URL = 'http://localhost:9000'


def build_ptb_dataset(conllu_files=[]):
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

    for i, f in enumerate(sents_list):
        sents_list[i] = [s[:, CONLLU_MASK] for s in f]

    # "Standard" train/dev/split for PTB
    train_list = [s for f in sents_list[2:22] for s in f]
    dev_list = sents_list[22]
    test_list = sents_list[23]

    train_list, word_counts = filter_and_count(train_list, filter_single=True)
    dev_list, _ = filter_and_count(dev_list, filter_single=False)
    test_list, _ = filter_and_count(test_list, filter_single=False)

    x2i, i2x = build_dicts(train_list)

    train_list = numericalize_sdp(train_list, x2i)
    dev_list = numericalize_sdp(dev_list, x2i)
    test_list = numericalize_sdp(test_list, x2i)

    data = {'train': train_list,
            'dev': dev_list,
            'test': test_list}

    return data, x2i, i2x, word_counts


def build_sdp_dataset(conllu_files: list, x2i=None):
    '''
        For building a dataset from conllu files in general, once the x2i dict
        has been constructed by build_ptb_dataset.
    '''
    data = {}

    for f in conllu_files:
        name = os.path.splitext(f)[0].split('/')[-1].lower()
        data[name] = conllu_to_sents(f)

    for name, sents in data.items():
        filtered, _ = filter_and_count([s[:, CONLLU_MASK] for s in sents], filter_single=False)
        data[name] = numericalize_sdp(filtered, x2i)

    return data


def build_ss_dataset(raw_sent_pairs, gs='', x2i=None):
    x2i, i2x = build_dicts(raw_sent_pairs, is_sdp=False)
    raw_sent_pairs = numericalize_ss(raw_sent_pairs, x2i)

    raw_targets = txt_to_sem_scores(gs) if gs else None
    
    sent_pairs = []
    targets = []
    if raw_targets != None:
        for s, t in zip(raw_sent_pairs, raw_targets):
            if t != -1.0:
                sent_pairs.append(s)
                targets.append(t)
        if len(targets) != len(sent_pairs):
            print('Mismatch between targets ({len(targets)}) and sents ({len(sent_pairs)})')
            raise Exception
    else:
        sent_pairs = raw_sent_pairs

    return {'sent_pairs': sent_pairs, 'targets': targets}


def txt_to_sem_scores(txt: str) -> list:
    with open(txt, 'r') as f:
        lines = f.readlines()
        sem_scores = [float(l.strip()) if l != '\n' else -1.0 for l in lines]

    return sem_scores


def build_cutoff_dicts(data_sorted: list) -> dict:
    '''
        inputs:
            data_sorted - list of np arrays (conllu-formatted sentences)

        returns:
            a dictionary i2c, keys are indices in sorted data, values are lists with 2 elements,
            the first index in data_sorted of a sentence of that length and the 
            (non-inclusive) final index
    '''
    i2c = dict()
    l2c = defaultdict(list)
    l2n = defaultdict(int)

    l_prev = data_sorted[0].shape[0]
    l_max = data_sorted[-1].shape[0]
    l2c[l_prev].append(0)
    l2n[l_prev] += 1
    for i, s in enumerate(data_sorted[1:], start=1):
        l = s.shape[0]
        l2n[l] += 1
        if l > l_prev:
            l2c[l_prev].append(i)
            l2c[l].append(i)
        l_prev = l
    l2c[l_max].append(len(data_sorted))

    for c in l2c.values():
        for i in range(c[0], c[1]):
            i2c[i] = c

    if len(i2c) != len(data_sorted):
        print(f'i2c {len(i2c)} != data_sorted {len(data_sorted)}')
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
    '''
        inputs:
            batch -
            paired -

        outputs:
            scores - a (b,1) tensor of 'scores' for the paired sentences, weights to be used in loss function
    '''
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


def length_to_results(data_sorted, l2c=None, device=None) -> dict:
    l2r = {}

    #for l, c in tqdm(l2c.items()):
    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building l2r', ncols=80):
        idxs = list(range(c[0], c[1]))
        t = torch.zeros(len(idxs), len(idxs))
        print(f'Tensor for length {l} has shape: {t.shape}')


        UAS_tensors = []
        LAS_tensors = []
        for chunk in idx_chunks(idxs, 100):
            s1_batch = []
            s2_batch = []
            for i, idx_i in enumerate(chunk):
                for j, idx_j in enumerate(chunk[i+1:]):
                    s1_batch.append(data_sorted[idx_i])
                    s2_batch.append(data_sorted[idx_j])
            
            results = get_syntactic_scores(
                    prepare_batch_sdp(s1_batch),
                    prepare_batch_sdp(s2_batch),
                    device=device)

            UAS_tensors.append(results['UAS'].flatten())
            LAS_tensors.append(results['LAS'].flatten())

        UAS_t = torch.cat(UAS_tensors, dim=0)
        LAS_t = torch.cat(LAS_tensors, dim=0)
        l2r[l] = {'UAS': UAS_t, 'LAS': LAS_t}

        #s1_batch = []
        #s2_batch = []
        #for i, idx_i in enumerate(idxs):
        #    for j, idx_j in enumerate(range(idx_i + 1, c[1])):
        #        s1_batch.append(data_sorted[idx_i])
        #        s2_batch.append(data_sorted[idx_j])
        #
        #print('Fetching syntactic scores')
        #results = get_syntactic_scores(
        #        prepare_batch_sdp(s1_batch),
        #        prepare_batch_sdp(s2_batch),
        #        device=device)

        #l2r[l] = results

    return l2r


def get_score_tensors(data_sorted, l2c=None, l2r=None, score_type=None, device=None) -> dict:
    l2t = {}
    num_duplicates = 0

    #for l, c in tqdm(l2c.items()):
    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building {score_type} l2t', ncols=80):
        idxs = list(range(c[0], c[1]))
        t = torch.zeros(len(idxs), len(idxs))

        #s1_batch = []
        #s2_batch = []
        #for i, idx_i in enumerate(idxs):
        #    for j, idx_j in enumerate(range(idx_i + 1, c[1])):
        #        s1_batch.append(data_sorted[idx_i])
        #        s2_batch.append(data_sorted[idx_j])
        #
        #results = get_syntactic_scores(
        #        prepare_batch_sdp(s1_batch),
        #        prepare_batch_sdp(s2_batch),
        #        score_type=score_type,
        #        device=device)
        
        scores = l2r[l][score_type]

        for i, idx_i in enumerate(idxs):
            for j, idx_j in enumerate(range(idx_i + 1, c[1])):
                t[i,j] = scores[i+j]
                if scores[i+j] == 1.0:
                    num_duplicates += 1

        #for i, idx_i in enumerate(idxs):
        #    si = prepare_batch_sdp([data_sorted[idx_i]])
        #    for j, idx_j in enumerate(range(idx_i+1, c[1])):
        #        sj = prepare_batch_sdp([data_sorted[idx_j]])
        #        t[i,j] = get_syntactic_scores(si, sj, score_type=score_type).flatten()
        #        #curr_pairwise[(i,j)] = get_syntactic_scores([s1], [s2], score_type='LAS').flatten()


        l2t[l] = t

    return l2t, num_duplicates


def sdp_data_loader_original(data, batch_size=1, shuffle_idx=False, custom_task=False):
    idx = list(range(len(data)))

    if custom_task:
        data_sorted = sorted(data, key = lambda s : s.shape[0])
        cutoff_dicts = build_cutoff_dicts(data_sorted)
        #l2n = cutoff_dicts['l2n']
        #i2c = cutoff_dicts['i2c']
        l2c = cutoff_dicts['l2c']

    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle
        
        if custom_task:
            paired_idx = get_paired_idx(idx, cutoff_dicts['i2c'])

            for chunk, chunk_p in zip(
                    idx_chunks(idx, batch_size),
                    idx_chunks(paired_idx, batch_size)):

                batch = [data_sorted[i] for i in chunk]
                paired = [data_sorted[i] for i in chunk_p]
                prepared_batch = prepare_batch_sdp(batch)
                prepared_paired = prepare_batch_sdp(paired)
                #scores = None  #XXX (b, 1) tensor
                results = get_syntactic_scores(prepared_batch, prepared_paired)  #XXX (b, 1) tensor
                scores = results['LAS']
                yield (prepared_batch, prepared_paired, scores)
        else:
            for chunk in idx_chunks(idx, batch_size):
                batch = [data[i] for i in chunk]
                yield prepare_batch_sdp(batch)


def sdp_data_loader_custom(data, batch_size=1):
    raise Exception
    idx = list(range(len(data)))

    #data_sorted = sorted(data, key = lambda s : s.shape[0])
    #cutoff_dicts = build_cutoff_dicts(data_sorted)
    ##l2n = cutoff_dicts['l2n']
    ##i2c = cutoff_dicts['i2c']
    #l2c = cutoff_dicts['l2c']


    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle

        #num_buckets = len(buckets)

        #for bucket in buckets:
        #    pass
            #from bucket rando-grab batch_size/num_buckets many indices
            #idx.extend(batch)

        paired_idx = get_paired_idx(idx, cutoff_dicts['i2c'])

        for chunk, chunk_p in zip(
                idx_chunks(idx, batch_size),
                idx_chunks(paired_idx, batch_size)):

            #batch = [data_sorted[i] for i in chunk]
            #paired = [data_sorted[i] for i in chunk_p]
            batch = [data[i] for i in chunk]
            paired = [data[i] for i in chunk_p]
            prepared_batch = prepare_batch_sdp(batch)
            prepared_paired = prepare_batch_sdp(paired)
            #scores = None  #XXX (b, 1) tensor
            results = get_syntactic_scores(prepared_batch, prepare_paired)  #XXX (b, 1) tensor
            scores = results['LAS']
            yield (prepared_batch, prepared_paired, scores)


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
    '''
        inputs:
            batch - 

        outputs:
            words - 
            pos -
            sent_lens - list of lengths (INCLUDES ROOT TOKEN)
            arc_targets -
            rel_targets -
    '''
    batch_size = len(batch)
    #batch_sorted = sorted(batch, key = lambda s: s.shape[0], reverse=True)
    sent_lens = torch.LongTensor([s.shape[0] for s in batch]) # Keep in mind, these lengths include ROOT token in each sentence
    #l_longest = sent_lens[0]
    l_longest = torch.max(sent_lens).item()

    words = torch.zeros((batch_size, l_longest)).long()
    pos = torch.zeros((batch_size, l_longest)).long()
    arc_targets = torch.LongTensor(batch_size, l_longest).fill_(-1)
    rel_targets = torch.LongTensor(batch_size, l_longest).fill_(-1)

    #for i, s in enumerate(batch_sorted):
    for i, s in enumerate(batch):
        # s shape (length, 4)

        dt = np.dtype(int)
        resized_np = np.zeros((l_longest, s.shape[1]), dtype=dt)
        resized_np[:s.shape[0]] = s
        words[i] = torch.LongTensor(resized_np[:,0])
        pos[i] = torch.LongTensor(resized_np[:,1])
        arc_targets[i] = torch.LongTensor(resized_np[:,2])
        rel_targets[i] = torch.LongTensor(resized_np[:,3])

    return {'words': words, 
            'pos' : pos, 
            'sent_lens' : sent_lens, 
            'arc_targets' : arc_targets, 
            'rel_targets' : rel_targets}


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
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)
    '''


    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()
        if lines[-1] != '\n':
            lines.append('\n') # So split_points works properly

    while(lines[0] == '\n'):
        lines.pop(0)

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

    exit()
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


def decode_sdp_sents(sents=[], i2x=None) -> list:
    i2w = i2x['word']
    i2p = i2x['pos']
    i2r = i2x['rel']

    decoded_sents = []
    for sent in sents:
        # sent is a (l,4) np array
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


def megabatch_breakdown(megabatch, minibatch_size, parser, device):
    '''
        inputs:
            megabatch - an unprepared megabatch (M many batches) of sentences
            batch_size - size of a minibatch

        outputs:
            s1 - list of orig. sentence instances
            mb_para2 - list of paraphrase instances
            mb_neg1 - list of neg sample instances
    '''
    mb_para1 = []
    mb_para2 = []
    
    for para1, para2 in megabatch:
        mb_para1.append(para1) # Does this allocate new memory?
        mb_para2.append(para2)

    minibatches_para1 = [mb_para1[i:i+minibatch_size] for i in range(0, len(mb_para1), minibatch_size)]
    minibatches_para2 = [mb_para2[i:i+minibatch_size] for i in range(0, len(mb_para2), minibatch_size)]

    if len(minibatches_para1) * minibatch_size != len(megabatch):
        raise Exception

    mb_para1_reps = [] # (megabatch_size, )
    mb_para2_reps = [] # (megabatch_size, )
    for b1, b2 in zip(minibatches_para1, minibatches_para2):
        w1, p1, sl1 = prepare_batch_ss(b1)
        w2, p2, sl2 = prepare_batch_ss(b2)
        sl1 = sl1.to(device)
        sl2 = sl2.to(device)
        b1_reps, _ = parser.BiLSTM(w1.to(device), p1.to(device), sl1)
        b2_reps, _ = parser.BiLSTM(w2.to(device), p2.to(device), sl2)
        b1_reps_avg = utils.average_hiddens(b1_reps, sl1)
        b2_reps_avg = utils.average_hiddens(b2_reps, sl2)
        mb_para1_reps.append(b1_reps_avg)
        mb_para2_reps.append(b2_reps_avg)

    # Stack all reps into torch tensors
    mb_para1_reps = torch.cat(mb_para1_reps)
    mb_para2_reps = torch.cat(mb_para2_reps)

    # Get negative samples with respect to mb_para1
    #mb_neg1 = get_negative_samps(mb_para1, megabatch_of_reps)
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
    negs = []

    reps = []
    sents = []
    #for para1, rep in zip(mb_para1, megabatch_of_reps): 
    for i, (para1, para2) in enumerate(megabatch):
    #for i in range(len(megabatch)):
        #(s1, _) = megabatch[i]
        #reps.append(megabatch_of_reps[i].cpu().numpy())
        reps.append(mb_para1_reps[i].cpu().numpy())
        reps.append(mb_para2_reps[i].cpu().numpy())
        sents.append(para1)
        sents.append(para2)

    dists = pdist(reps, 'cosine') # cosine distance, as (1 - normalized inner product)
    dists = squareform(dists) # Symmetric 2-D matrix of pairwise distances

    # Don't risk pairing a sentence with itself
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
    mb_neg2 = []
    #for idx in neg_idxs:
    for idx in range(len(megabatch)):
        #neg = sents[idx]
        #negs.append(neg)
        neg1 = sents[neg_idxs[2*idx]]
        neg2 = sents[neg_idxs[(2*idx)+1]]
        mb_neg1.append(neg1)
        mb_neg2.append(neg2)

    return mb_neg1, mb_neg2


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


def sdp_corpus_stats(data, stats_pkl='../data/sdp_corpus_stats.pkl', stats_readable ='../data/readable_stats.txt', device=None):
    stats = {}
    data_sorted = sorted(data, key = lambda s : s.shape[0])
    bucket_dicts = build_bucket_dicts(data_sorted)
    
    l2n = bucket_dicts['l2n']
    i2c = bucket_dicts['i2c']
    l2c = bucket_dicts['l2c']

    for l, n in l2n.items():
        if n < 2:
            l2c.pop(l)

    print(len(l2c))

    if not os.path.exists(stats_pkl):
        l2r = length_to_results(data_sorted, l2c=l2c, device=device)

        l2t_UAS, ud = get_score_tensors(data_sorted, l2c=l2c, l2r=l2r, score_type='UAS', device=device)
        l2t_LAS, ld = get_score_tensors(data_sorted, l2c=l2c, l2r=l2r, score_type='LAS', device=device)

        avgs_UAS, ou = l2t_to_l2avg(l2t_UAS)
        avgs_LAS, ol = l2t_to_l2avg(l2t_LAS)

        stats['l2t_UAS'] = l2t_UAS
        stats['l2t_LAS'] = l2t_LAS
        stats['avgs_UAS'] = avgs_UAS
        stats['avgs_LAS'] = avgs_LAS
        stats['ou'] = ou
        stats['ol'] = ol
        stats['ud'] = ud
        stats['ld'] = ld

        with open(stats_pkl, 'wb') as f:
            pickle.dump(stats, f)
    else:
        with open(stats_pkl, 'rb') as f:
            stats = pickle.load(f)

    l2t_UAS = stats['l2t_UAS']
    l2t_LAS = stats['l2t_LAS'] 
    avgs_UAS = stats['avgs_UAS'] 
    avgs_LAS = stats['avgs_LAS'] 
    ou = stats['ou']
    ol = stats['ol']
    ud = stats['ud']
    ld = stats['ld']

    print('UAS averages: ', avgs_UAS)
    print('LAS averages: ', avgs_LAS)

    print('UAS overall: ', ou)
    print('LAS overall: ', ol)

    unique_len_counter = 0
    for n in l2n.values():
        if n <= 1:
            unique_len_counter += 1

    print('Unique lengths: ', unique_len_counter)

    with open(stats_readable, 'w') as f:
        f.write('Sentence length\tNum of length\tAvg UAS\t Avg LAS\n')
        for l, c in l2c.items():
            n = len(range(c[0], c[1]))
            f.write('{:10}\t{:10}\t{:10.3f}\t{:10.3f}\n'.format(
                l,
                n,
                avgs_UAS[l],
                avgs_LAS[l]))

        f.write(f'\nUnique lengths: {unique_len_counter}')
        f.write(f'\nNum dupes (by UAS): {ud}')
        f.write(f'\nNum dupes (by LAS): {ld}')


def l2t_to_l2avg(l2t):
    l2avg = {}
    avg_list = []

    for l, t in l2t.items():
        #nonzeros = t.flatten().index_select(0, t.nonzero().flatten())
        #avg = torch.mean(nonzeros).item()
        divisor = (t.shape[0] * (t.shape[0] - 1)) / 2
        avg = (t.sum() / divisor).item()
        avg_list.append(avg)
        l2avg[l] = avg
    
    overall_avg = torch.tensor(avg_list).mean().item()

    return l2avg, overall_avg



