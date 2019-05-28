import logging
import os
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
from collections import Counter

import data_utils

LOG_DIR = '../log'
DATA_DIR = '../data'

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
STS_DIR = f'{DATA_DIR}/sts'
STS_INPUT = os.path.join(STS_DIR, 'input')
STS_GS = os.path.join(STS_DIR, 'gs')
DEP_DIR = f'{CORPORA_DIR}/wsj/dependencies'
#BROWN_DIR = f'{CORPORA_DIR}/brown/dependencies'
BROWN_DIR = '../data/brown'
PARANMT_DIR = os.path.join(DATA_DIR, 'paranmt_5m')

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'preprocess.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


class DataPaths:
    def __init__(self, filtered=False, glove_d=None):
        fs = 'filtered' if filtered else 'unfiltered' # Filtered status

        voc = 'ptbvocab'
        if glove_d is not None:
            voc = f'glove{glove_d}vocab'
    
        # Directories
        sdp_data_dir = os.path.join(DATA_DIR, 'sdp_processed')
        ss_data_dir = os.path.join(DATA_DIR, 'ss_processed')
        syn_data_dir = os.path.join(DATA_DIR, 'syn_processed')
        if not os.path.isdir(sdp_data_dir):
            os.mkdir(sdp_data_dir)
        if not os.path.isdir(ss_data_dir):
            os.mkdir(ss_data_dir)
        if not os.path.isdir(syn_data_dir):
            os.mkdir(syn_data_dir)
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'pkl')):
            os.mkdir(os.path.join(PARANMT_DIR, 'pkl'))
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'tagged')):
            os.mkdir(os.path.join(PARANMT_DIR, 'tagged'))
        if not os.path.isdir(os.path.join(STS_DIR, 'tagged')):
            os.mkdir(os.path.join(STS_DIR, 'tagged'))
    
        self.ptb_vocabs = os.path.join(sdp_data_dir, f'vocabs_{fs}.pkl')
        self.data_ptb = os.path.join(sdp_data_dir, f'data_ptb_{fs}_{voc}.pkl')
        self.data_brown = os.path.join(sdp_data_dir, f'data_brown_{fs}_{voc}.pkl')


        self.cutoff_dicts = os.path.join(syn_data_dir, f'syn_cutoff_dicts_{fs}_{voc}.pkl')
        self.l2p = os.path.join(syn_data_dir, f'syn_l2p_{fs}_{voc}.pkl')
        self.syn_data = os.path.join(syn_data_dir, f'syn_data_{fs}_{voc}.pkl')
            
        self.glove_data = os.path.join(DATA_DIR, 'glove', f'glove_{glove_d}_data.pkl')
        self.glove_w2v = os.path.join(DATA_DIR, 'glove', f'glove_{glove_d}_w2v.pkl')
        self.ss_train_base = os.path.join(PARANMT_DIR, 'pkl', f'{fs}_{voc}_')
        self.ss_test = os.path.join(ss_data_dir, f'ss_test_{fs}_{voc}.pkl')
        #self.paranmt_counts = os.path.join(ss_data_dir, f'paranmt_counts.pkl')
        #self.sl999_data = os.path.join(ss_data_dir, f'sl999_data.pkl')
        #self.sl999_w2v = os.path.join(ss_data_dir, f'sl999_w2v.pkl')
    

def preprocess(args):
    paths = DataPaths(filtered=args.filter, glove_d=args.glove_d)

    x2i, i2x = {}, {}
    if os.path.exists(paths.ptb_vocabs):
        with open(paths.ptb_vocabs, 'rb') as f:
            x2i, i2x = pickle.load(f)

    if args.pretrained_voc:
        #path = paths.sl999_data if args.sl999 else paths.glove_data
        path = paths.glove_data
        with open(path, 'rb') as pkl:
            embedding_data = pickle.load(pkl)

        x2i['word'] = embedding_data['w2i']
        i2x['word'] = embedding_data['i2w']


    if args.sdp != []:
        if 'ptb' in args.sdp:
            log.info(f'Initializing Penn Treebank WSJ data (including vocabs).')
            ptb_conllus = sorted(
                    [os.path.join(DEP_DIR, f) for f in os.listdir(DEP_DIR)])
            raw_data_ptb, x2i_ptb, i2x_ptb, word_counts = data_utils.build_ptb_dataset(
                    ptb_conllus, 
                    filter_sents=args.filter)
            with open(paths.ptb_vocabs, 'wb') as f:
                pickle.dump((x2i_ptb, i2x_ptb), f)
            x2i.update(x2i_ptb)
            i2x.update(i2x_ptb)
            if args.pretrained_voc:
                x2i['word'] = embedding_data['w2i']
                i2x['word'] = embedding_data['i2w']

            data_ptb = {}
            for split, raw_data in raw_data_ptb.items():
                data_ptb[split] = data_utils.numericalize_sdp(raw_data, x2i)
            with open(paths.data_ptb, 'wb') as f:
                pickle.dump((data_ptb, word_counts), f)

        if 'brown' in args.sdp:
            log.info(f'Initializing Brown corpus data.')
            brown_conllus = [os.path.join(BROWN_DIR, f) for f in os.listdir(BROWN_DIR)]
            data_brown = data_utils.build_sdp_dataset(
                    brown_conllus, 
                    x2i=x2i, 
                    filter_sents=args.filter)
            with open(paths.data_brown, 'wb') as f:
                pickle.dump(data_brown, f)


    #if args.paranmt_counts:
    #    word_counts = Counter()
    #    chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
    #    for chunk in tqdm(chunks_txt, ascii=True, desc=f'Getting word counts for ParaNMT corpus', ncols=80):
    #        raw_sents_path = os.path.join(PARANMT_DIR, 'tagged', f'{os.path.splitext(chunk)[0]}-tagged.pkl')
    #        with open(raw_sents_path, 'rb') as pkl:
    #            raw_sent_pairs = pickle.load(pkl)

    #        flattened_raw = []
    #        for s1, s2 in raw_sent_pairs:
    #            flattened_raw.append(s1)
    #            flattened_raw.append(s2)
    #        wc_chunk = data_utils.get_word_counts(flattened_raw)
    #        word_counts.update(wc_chunk)

    #    with open(paths.paranmt_counts, 'wb') as pkl:
    #        pickle.dump(word_counts, pkl)


    #if args.sl999 and not args.pretrained_voc:
    #    w2v = data_utils.build_pretrained_w2v(word_v_file='../data/paranmt_5m/paragram_300_sl999_top100k.txt')
    #    with open(paths.sl999_w2v, 'wb') as pkl:
    #        pickle.dump(w2v, pkl)
    #    with open(paths.paranmt_counts, 'rb') as pkl:
    #        word_counts = pickle.load(pkl)

    #    n_removed = 0
    #    w2v_cleaned = {}
    #    for w, v in w2v.items():
    #        if word_counts[w] != 0:
    #            w2v_cleaned[w] = v
    #        else:
    #            n_removed += 1
    #    print(f'Removed {n_removed} words from w2v.') 
    #    
    #    sl999_data = data_utils.build_embedding_data(w2v_cleaned)
    #    with open(paths.sl999_data, 'wb') as pkl:
    #        pickle.dump(sl999_data, pkl)


    if args.glove_d and not args.pretrained_voc:
        w2v = data_utils.build_pretrained_w2v(word_v_file=f'../data/glove/glove.6B.{args.glove_d}d.top100k.txt')
        with open(paths.glove_w2v, 'wb') as pkl:
            pickle.dump(w2v, pkl)
        with open(paths.paranmt_counts, 'rb') as pkl:
            word_counts = pickle.load(pkl)

        n_removed = 0
        w2v_cleaned = {}
        for w, v in w2v.items():
            if word_counts[w] != 0:
                w2v_cleaned[w] = v
            else:
                n_removed += 1
        print(f'Removed {n_removed} words from w2v.') 

        glove_data = data_utils.build_embedding_data(w2v_cleaned)
        with open(paths.glove_data, 'wb') as pkl:
            pickle.dump(glove_data, pkl)


    if 'train' in args.ss:
        log.info(f'Initializing semantic similarity train data.')

        chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
        for chunk in chunks_txt:
            print(f'Processing chunk {chunk}')
            raw_sents_path = os.path.join(PARANMT_DIR, 'tagged', f'{os.path.splitext(chunk)[0]}-tagged.pkl')
            if args.pos_only:
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(PARANMT_DIR, 'txt', chunk))

                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
            else:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)

                train_path_chunk = paths.ss_train_base + f'{os.path.splitext(chunk)[0]}.pkl'
                train_ss_chunk = data_utils.build_ss_dataset(
                        raw_sent_pairs, 
                        gs='', 
                        x2i=x2i,
                        filter_sents=args.filter)
                with open(train_path_chunk, 'wb') as pkl:
                    pickle.dump(train_ss_chunk, pkl)


    if 'test' in args.ss:
        log.info(f'Initializing semantic similarity test data.')

        test_ss = {}
        years = os.listdir(STS_INPUT)
        for year in years:
            raw_sents_path = os.path.join(STS_DIR, 'tagged', f'{year}-tagged.pkl')
            if args.pos_only:
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(STS_INPUT, year))
                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
            else:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)
                test_ss[year] = data_utils.build_ss_dataset(
                        raw_sent_pairs,
                        gs=os.path.join(STS_GS, year),
                        x2i=x2i,
                        filter_sents=args.filter)
        with open(paths.ss_test, 'wb') as pkl:
            pickle.dump(test_ss, pkl)



    if args.syn:
        with open(paths.data_ptb, 'rb') as pkl:
            data_ptb = pickle.load(pkl)
        sents_sorted = sorted(data_ptb['train'], key=lambda sent: sent.shape[0])

        if 'cutoffs' in args.syn:
            cutoff_dicts = data_utils.build_cutoff_dicts(sents_sorted)
            with open(paths.cutoff_dicts, 'wb') as pkl:
                pickle.dump(cutoff_dicts, pkl)
        else:
            with open(paths.cutoff_dicts, 'rb') as pkl:
                cutoff_dicts = pickle.load(pkl)

        if 'pairs' in args.syn:
            l2p = data_utils.build_l2p(sents_sorted, l2c=cutoff_dicts['l2c'])
            with open(paths.l2p, 'wb') as pkl:
                pickle.dump(l2p, pkl)
        else:
            with open(paths.l2p, 'rb') as pkl:
                l2p = pickle.load(pkl)

        if 'buckets' in args.syn:
            l2b = data_utils.build_l2b(
                    sents_sorted, 
                    l2p=l2p, 
                    granularity=args.granularity, 
                    score_type=args.score_type)

            syn_data = {'l2n': cutoff_dicts['l2n'], 'l2b': l2b}
            with open(paths.syn_data, 'wb') as pkl:
                pickle.dump(syn_data, pkl)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-ss', help='Initialize a part (or all) of the semantic similarity data.', dest='ss', nargs='*', type=str, default=[])
    parser.add_argument('--pos', action='store_true', dest='pos_only', default=False)

    parser.add_argument('-sdp', help='Initialize a part (or all) of the syntactic parsing data.', dest='sdp', nargs='*', type=str, default=[])

    parser.add_argument('-syn', help='Initialize data for special syntactic task.', action='store_true', dest='syn', default=False)
    parser.add_argument('-gr', help='What should be the bucketing granularity for sentence scores?', type=float, dest='granularity', default=0.25)
    parser.add_argument('-st', help='Score type', type=str, dest='score_type', default='LAS')

    pretrained_emb = parser.add_mutually_exclusive_group()
    #pretrained_emb.add_argument('-sl999', action='store_true', help='Initialize data corresponding to sl999 word embeddings.', dest='sl999', default=False)
    #parser.add_argument('-counts', action='store_true', help='Get counts of word ocurrences in ParaNMT corpus.', dest='paranmt_counts', default=False)
    pretrained_emb.add_argument('-g', help='Initialize data corresponding to glove word embeddings.', dest='glove_d', type=int, default=None)
    parser.add_argument('--pevoc', action='store_true', help='Shall we use the word vocabulary derived from the chosen pretrained embeddings?', dest='pretrained_voc', default=False)

    parser.add_argument('-f', help='Should sentences be filtered?', action='store_true', dest='filter', default=False)

    args = parser.parse_args()

    preprocess(args)

