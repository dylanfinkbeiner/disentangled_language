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
    def __init__(self, filtered=None):
        fs = 'filtered' if filtered else 'unfiltered' # Filtered status
    
        # Directories
        sdp_data_dir = os.path.join(DATA_DIR, 'sdp_processed')
        ss_data_dir = os.path.join(DATA_DIR, 'ss_processed')
        if not os.path.isdir(sdp_data_dir):
            os.mkdir(sdp_data_dir)
        if not os.path.isdir(ss_data_dir):
            os.mkdir(ss_data_dir)
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'pkl')):
            os.mkdir(os.path.join(PARANMT_DIR, 'pkl'))
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'tagged')):
            os.mkdir(os.path.join(PARANMT_DIR, 'tagged'))
        if not os.path.isdir(os.path.join(STS_DIR, 'tagged')):
            os.mkdir(os.path.join(STS_DIR, 'tagged'))
    
        self.vocabs = os.path.join(sdp_data_dir, f'vocabs_{fs}.pkl')
        self.data_ptb = os.path.join(sdp_data_dir, f'data_ptb_{fs}.pkl')
        self.data_brown = os.path.join(sdp_data_dir, f'data_brown_{fs}.pkl')
            
        self.ss_train_base = os.path.join(PARANMT_DIR, 'pkl', f'{fs}_')
        self.paranmt_counts = os.path.join(ss_data_dir, f'paranmt_counts.pkl')
        self.sl999_data = os.path.join(ss_data_dir, f'sl999_data.pkl')
        self.sl999_w2v = os.path.join(ss_data_dir, f'sl999_w2v.pkl')
        self.ss_dev = os.path.join(ss_data_dir, f'ss_dev_{fs}.pkl')
        self.ss_test = os.path.join(ss_data_dir, f'ss_test_{fs}.pkl')
    

def preprocess(args):
    fs = 'filtered' if args.filter else 'unfiltered' # Filtered status

    paths = DataPaths()

    if args.sdp != []:
        if 'ptb' in args.sdp:
            log.info(f'Initializing Penn Treebank WSJ data (including vocabs).')
            ptb_conllus = sorted(
                    [os.path.join(DEP_DIR, f) for f in os.listdir(DEP_DIR)])
            data_ptb, x2i, i2x, word_counts = data_utils.build_ptb_dataset(
                    ptb_conllus, 
                    filter_sents=args.filter)
            with open(paths.vocabs, 'wb') as f:
                pickle.dump((x2i, i2x), f)
            with open(paths.data_ptb, 'wb') as f:
                pickle.dump((data_ptb, word_counts), f)

        if 'brown' in args.sdp:
            log.info(f'Initializing Brown corpus data.')
            brown_conllus = [os.path.join(BROWN_DIR, f) for f in os.listdir(BROWN_DIR)]
            with open(paths.vocabs, 'rb') as f:
                x2i, i2x = pickle.load(f)
            data_brown = data_utils.build_sdp_dataset(
                    brown_conllus, 
                    x2i=x2i, 
                    filter_sents=args.filter)
            with open(paths.data_brown, 'wb') as f:
                pickle.dump(data_brown, f)
    else: # Must load dictionaries for initializing other datasets
        with open(paths.vocabs, 'rb') as f:
            x2i, i2x = pickle.load(f)


    if args.paranmt_counts:
        word_counts = Counter()
        chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
        for chunk in tqdm(chunks_txt, ascii=True, desc=f'Getting word counts for ParaNMT corpus', ncols=80):
            raw_sents_path = os.path.join(PARANMT_DIR, 'tagged', f'{os.path.splitext(chunk)[0]}-tagged.pkl')
            with open(raw_sents_path, 'rb') as pkl:
                raw_sent_pairs = pickle.load(pkl)

            flattened_raw = []
            for s1, s2 in raw_sent_pairs:
                flattened_raw.append(s1)
                flattened_raw.append(s2)
            wc_chunk = data_utils.get_word_counts(flattened_raw)
            word_counts.update(wc_chunk)

        with open(paths.paranmt_counts, 'wb') as pkl:
            pickle.dump(word_counts, pkl)


    if args.sl999:
        w2v = data_utils.build_sl999_w2v(word_v_file='../data/paranmt_5m/paragram_300_sl999_top100k.txt')
        with open(paths.sl999_w2v, 'wb') as pkl:
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
        
        sl999_data = data_utils.build_sl999_data(w2v_cleaned)
        with open(paths.sl999_data, 'wb') as pkl:
            pickle.dump(sl999_data, pkl)


    if args.use_paragram:
        with open(paths.sl999_data, 'rb') as pkl:
            sl999_data = pickle.load(pkl)

            x2i['word'] = sl999_data['w2i']
            i2x['word'] = sl999_data['i2w']


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


    if 'dev' in args.ss:
        log.info(f'Initializing semantic similarity dev data.')
        raw_sents_path = os.path.join(STS_DIR, 'tagged', '2017-tagged.pkl')

        if args.pos_only:
            raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(STS_INPUT, '2017'))

            with open(raw_sents_path, 'wb') as pkl:
                pickle.dump(raw_sent_pairs, pkl)
        else:
            with open(raw_sents_path, 'rb') as pkl:
                raw_sent_pairs = pickle.load(pkl)
            dev_ss = data_utils.build_ss_dataset(
                    raw_sent_pairs,
                    gs=os.path.join(STS_GS, '2017'),
                    x2i=x2i,
                    filter_sents=args.filter)
            with open(paths.ss_dev, 'wb') as pkl:
                pickle.dump(dev_ss, pkl)

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




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-ss', help='Initialize a part (or all) of the semantic similarity data.', dest='ss', nargs='*', type=str, default=[])
    parser.add_argument('--pos', action='store_true', dest='pos_only', default=False)
    parser.add_argument('-sdp', help='Initialize a part (or all) of the syntactic parsing data.', dest='sdp', nargs='*', type=str, default=[])
    parser.add_argument('-sl999', action='store_true', help='Initialize data corresponding to sl999 word embeddings.', dest='sl999', default=False)
    parser.add_argument('-counts', action='store_true', help='Get counts of word ocurrences in ParaNMT corpus.', dest='paranmt_counts', default=False)
    parser.add_argument('--paragram', action='store_true', help='Shall we use the word vocabulary derived from the paragram word vectors?', dest='use_paragram', default=False)

    parser.add_argument('-f', help='Should sentences be filtered?', action='store_true', dest='filter', default=False)

    args = parser.parse_args()

    preprocess(args)

