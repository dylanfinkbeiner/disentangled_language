import os
import sys
from collections import defaultdict
import pickle
from scipy.stats import pearsonr
import numpy as np

MODELS_DIR = '../experiments/'


def params_dict_from_txt(txt):
    params = {}
    with open(txt, 'r') as f:
        lines = f.readlines()

    for l in lines:
        split_l = l.split(' ')
        if split_l[2] == 'False':
            split_l[2] = False
        params[split_l[0]] = split_l[2]

    return params


def params_dict_from_pkl(pkl_path):
    params = {}
    with open(pkl_path, 'rb') as pkl:
        params = pickle.load(pkl)

    return params


def get_eval_data(name):
    eval_data = {}
    os.path.join(MODELS_DIR, name, '')

    exp_files = os.listdir(os.path.join(MODELS_DIR, name))
    eval_file = ''
    for f in exp_files:
        if os.path.splitext(f)[-1] == '.pkl' and f[:10] == 'evaluation':
            eval_file = f

    if eval_file != '':
        path = os.path.join(MODELS_DIR, name, eval_file)
        with open(path, 'rb') as pkl:
            eval_data = pickle.load(pkl)

    return eval_data


def get_models():
    models = defaultdict(dict)

    names = os.listdir(MODELS_DIR)

    #for name in names:
    #    models[name]['params'] = params_dict_from_txt(
    #            os.path.join(MODELS_DIR, name, 'parameters.txt'))
    for name in names:
        models[name]['params'] = params_dict_from_pkl(
                os.path.join(MODELS_DIR, name, 'parameters.pkl'))
    models = dict(models)

    for name in names:
        eval_data = get_eval_data(name)
        if len(eval_data) > 0:
            models[name]['eval_data'] = get_eval_data(name)
        else:
            models.pop(name)


    return models


def git(l, p):
    result = [m['params'][p] for m in l]
    return result


def main():
    global MODELS_DIR
    folder = str(sys.argv[1]) if len(sys.argv) > 1 else 'TUNING'
    MODELS_DIR += folder

    models = get_models()
    n_mods = len(models)

    mods = list(models.values())
    #mods = sorted(mods, key=lambda m : m['params']['alpha'])
    params = list(mods[0]['params'].keys())
    for p in list(params):
        try:
            f = float(mods[0]['params'][p])
        except Exception:
            params.remove(p)
    mods = sorted(mods, key=lambda m : m['params']['alpha'])

    # SDP
    sdpR = {}
    corw = {}
    corwo = {}
    try:
        #mods_syn = sorted(mods, key=lambda m : m['eval_data']['sdp']['ptb_dev'][1])
        mods_syn = mods
        #wpos = [m for m in models.values() if int(m['params']['pe']) != 0]
        #wopos = [m for m in models.values() if int(m['params']['pe']) == 0]
        #wpos = sorted(wpos, key=lambda m : m['eval_data']['sdp']['ptb_dev'][1])
        #wopos = sorted(wopos, key=lambda m : m['eval_data']['sdp']['ptb_dev'][1])
        LAS = [m['eval_data']['sdp']['ptb_dev'][1] for m in mods_syn]
        #LAS_w = [m['eval_data']['sdp']['ptb_dev'][1] for m in wpos]
        #LAS_wo = [m['eval_data']['sdp']['ptb_dev'][1] for m in wopos]

        sdpR = R_dict(params, models=mods_syn, values=LAS)
        #corw[p] = get_corr(param=p, models=wpos, LAS=LAS_w)
        #corwo[p] = get_corr(param=p, models=wopos, LAS=LAS_wo)
    except Exception:
        print('Missing/no syntactic dependency parsing evaluation data.')

    # Supertagging
    stagR = {}
    try:
        mods_stag = sorted(mods, key=lambda m : m['eval_data']['stag']['dev'])
        acc = [np.mean(list(m['eval_data']['stag'].values())) for m in mods_stag]

        stagR = R_dict(params, models=mods_stag, values=acc)
    except Exception:
        print('Missing/no supertagging evaluation data.')

    # Semantics
    semR = {}
    try:
        mods_sem = sorted(mods, key=lambda m : m['eval_data']['sem']['average'])
        avg = [m['eval_data']['sem']['average'] for m in mods_sem]

        semR = R_dict(params, models=mods_sem, values=avg)
    except Exception:
        print('Missing/no SemEval results.')


    # Brown
    try:
        #mods_brown = sorted(mods, key=lambda m : m['eval_data']['sdp']['ptb_dev'][1])
        mods_brown = mods
        LASb = [m['eval_data']['sdp']['brown_cf'][1] for m in mods_brown]
        UASb = [m['eval_data']['sdp']['brown_cf'][0] for m in mods_brown]
    except Exception:
        print('Missing/no Brown evaluation data.')


    breakpoint()


def R_dict(params, models=None, values=None):
    d = {}
    for p in params:
        try:
            d[p] = get_corr(param=p, models=models, values=values)

        except Exception:
            continue

    return d


def get_corr(param=None, models=None, values=None):
    try:
        ps = [float(m['params'][param]) for m in models]
        R, _ = pearsonr(ps, values)
        return R
    except ValueError:
        print(f'Cannot get correlation for: {param}')
        return None

def stats(vals : list):
    mean = np.mean(vals)
    std = np.std(vals)
    return (mean, std)

def line(d : dict):
    for k, v in d.items():
        if not np.isnan(v):
            print(f'{k} : {v}')


if __name__ == '__main__':
    main()
