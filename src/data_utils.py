import random
from random import shuffle
import string

from collections import defaultdict, Counter
import numpy as np
from nltk.parse import CoreNLPParser
from scipy.spatial.distance import pdist, squareform
import torch

import utils

UNK_TOKEN = '<unk>'
ROOT_TOKEN = '<root>'
PAD_TOKEN = '<pad>'

CONLLU_MASK = [1, 4, 6, 7]  # [word, pos, head, rel]
CORENLP_URL = 'http://localhost:9001'


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
    pairwise_scores


def txt_to_scores(txt: str):

    with open(txt, 'r') as f:
        lines = f.read_lines()

        [float(line) for line in lines]


def build_ss_dataset(paranmt_file='', scores_file='', x2i=None):
    sents_list = paraphrase_to_sents(paranmt_file)
    sents_list = numericalize_ss(sents_list, x2i)

    targets = txt_to_scores(scores_file) if scores_file else None
    
    data = []
    for s, t in zip(sents_list, targets):
        if t == '\n':
            sents_list.pop(s)

    if len(targets) != len(sents_list):
        print('Mismatch between targets ({len(targets)}) and sents ({len(sents_list)})')
        raise Exception

    return sents_list, targets


def build_bucket_dicts(data_sorted: list) -> dict:
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

    for cutoffs in l2c.values():
        for i in range(cutoffs[0], cutoffs[1]):
            i2c[i] = cutoffs

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
        unique_length = c[0] == c[1] - 1
        while (paired_i == i) and not unique_length:
            paired_i = random.randrange(c[0], c[1])
        paired_idx.append(paired_i)

    return paired_idx


def get_scores(batch, paired, score_type=None):
    '''
        inputs:
            batch -
            paired -

        outputs:
            scores - a (b,1) tensor of 'scores' for the paired sentences, weights to be used in loss function
    '''
    results = utils.attachment_scoring(
            arc_preds=batch['arc_targets'], 
            rel_preds=batch['rel_targets'], 
            arc_targets=paired['arc_targets'], 
            rel_targets=paired['rel_targets'], 
            sent_lens=batch['sent_lens'], 
            include_root=False,
            keep_dim=True)

    return results[score_type] # (b, 1)


def build_buckets(data_sorted, l2c=None, i2c=None, score_type=None) -> dict:
    buckets = {}

    for length in l2c.keys():
        curr_pairwise = defaultdict(int)

        c = l2c[length]
        for i in range(c[0], c[1]):
            s1 = data_sorted[i]
            for j in range(i+1, c[1]):
                s2 = data_sorted[j]
                curr_pairwise[(i,j)] = get_scores([s1], [s2], score_type='LAS')

        buckets[length] = dict(curr_pairwise)

    return buckets


def sdp_data_loader(data, batch_size=1, shuffle_idx=False, custom_task=False):
    idx = list(range(len(data)))

    if custom_task:
        data_sorted = sorted(data, key = lambda s : s.shape[0])
        bucket_dicts = build_bucket_dicts(data_sorted)
        l2n = bucket_dicts['l2n']
        cutoffs = bucket_dicts['i2c']

        l2c = bucket_dicts['l2c']
        print('Num lengths: ', len(l2c))

        print(l2c.keys())
        print(len(cutoffs))
        print(l2n.items())

        k = 0
        for v in l2n.values():
            k += v

        print(f'k: {k} , len: {len(data_sorted)}')

        exit()
    

    while True:
        if shuffle_idx:
            shuffle(idx) # In-place shuffle

        if custom_task:
            paired_idx = get_paired_idx(idx, cutoffs)

            for chunk, chunk_p in zip(
                    idx_chunks(idx, batch_size), 
                    idx_chunks(paired_idx, batch_size)):

                batch = [data_sorted[i] for i in chunk]
                paired = [data_sorted[i] for i in chunk_p]
                prepared_batch = prepare_batch_sdp(batch)
                prepared_paired = prepare_batch_sdp(paired)
                yield (prepared_batch,
                        prepared_paired, 
                        get_scores(prepared_batch, prepared_paired, score_type='LAS'))
        else:
            for chunk in idx_chunks(idx, batch_size):
                batch = [data[i] for i in chunk]
                yield prepare_batch_sdp(batch)


def ss_data_loader(data, batch_size=None):
    '''
        inputs:
            data - the full Python list of pairs of numericalized sentences (np arrays)
            batch_size - batch size

        yields:
            chunk - list of indices representing a minibatch
    '''
    idx = list(range(len(data)))
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
    batch_sorted = sorted(batch, key = lambda s: s.shape[0], reverse=True)
    sent_lens = torch.LongTensor([s.shape[0] for s in batch_sorted]) # Keep in mind, these lengths include ROOT token in each sentence
    length_longest = sent_lens[0]

    words = torch.zeros((batch_size, length_longest)).long()
    pos = torch.zeros((batch_size, length_longest)).long()
    arc_targets = torch.LongTensor(batch_size, length_longest).fill_(-1)
    rel_targets = torch.LongTensor(batch_size, length_longest).fill_(-1)

    for i, s in enumerate(batch_sorted):
        for j, _ in enumerate(s):
            '''
            Casting as ints because for some stupid reason
            you cannot set a value in torch long tensor using
            numpy's 64 bit ints
            '''
            words[i,j] = int(s[j,0])
            pos[i,j] = int(s[j,1])
            arc_targets[i,j] = int(s[j,2])
            rel_targets[i,j] = int(s[j,3])

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


def conllu_to_sents(f:str):
    '''
    inputs:
        f - filename of conllu file

    outputs:
        sents_list - a list of np arrays with shape (#words-in-sentence, 4)
    '''


    with open(f, 'r') as conllu_file:
        lines = conllu_file.readlines()

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


def get_triplets(megabatch, minibatch_size, parser, device):
    '''
        inputs:
            megabatch - an unprepared megabatch (M many batches) of sentences
            batch_size - size of a minibatch

        outputs:
            s1 - list of orig. sentence instances
            s2 - list of paraphrase instances
            negs - list of neg sample instances
    '''
    s1 = []
    s2 = []
    
    for mini in megabatch:
        s1.append(mini[0]) # Does this allocate new memory?
        s2.append(mini[1])

    minibatches = [s1[i:i + minibatch_size] for i in range(0, len(s1), minibatch_size)]

    megabatch_of_reps = [] # (megabatch_size, )
    for m in minibatches:
        words, pos, sent_lens = prepare_batch_ss(m)
        sent_lens = sent_lens.to(device)

        m_reps, _ = parser.BiLSTM(words.to(device), pos.to(device), sent_lens)
        megabatch_of_reps.append(utils.average_hiddens(m_reps, sent_lens))

    megabatch_of_reps = torch.cat(megabatch_of_reps)

    negs = get_negative_samps(megabatch, megabatch_of_reps)

    return s1, s2, negs


def get_negative_samps(megabatch, megabatch_of_reps):
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
    for i in range(len(megabatch)):
        (s1, _) = megabatch[i]
        reps.append(megabatch_of_reps[i].cpu().numpy())
        sents.append(s1)

    arr = pdist(reps, 'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i] = 0

    arr = np.argmax(arr, axis=1)

    for i in range(len(megabatch)):
        t = None
        t = sents[arr[i]]

        negs.append(t)

    return negs


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
