import subprocess
import math
import sys

import numpy as np


def main():
    #name_prefix = input('Prefix for model names: ').strip()

    #n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    #n_iter =  10 #x
    #n_iter =  4
    n_iter =  5

    #name_prefix = ['zeroUnk'] * n_iter
    #name_prefix = ['semUnifDrop'] * n_iter
    #name_prefix = ['semLayerDrop'] * n_iter
    #name_prefix = ['moreUnifDrop'] * n_iter
    #name_prefix = ['justAdvStag'] * n_iter
    #name_prefix = ['advStag2'] * n_iter
    #name_prefix = ['advStag3'] * n_iter
    #name_prefix = ['moreAdvStag'] * n_iter
    #name_prefix = ['zerokunktrials'] * (n_iter // 2) + ['randunktrials'] * (n_iter // 2)
    name_prefix = ['lrpos_trial']


    # Ranges
    synl_r = [1]
    #synl_r = [1, 2]
    #synl_r = [2, 3]
    #synl_r = [2] # x
    #synl_r = [2] 
    #finl_r = [1, 2, 3]
    #finl_r = [1]
    #finl_r = [1,2] #XXX
    finl_r = [2]

    #finh_r = [300, 400, 500, 600]
    #finh_r = [400, 500] #XXX
    finh_r = [400]
    #finh_r = [500]
    #frac_syn_r = [1/8 * x for x in range(4, 6)]
    frac_syn_r = [1]

    #bs_r = logrange_list(20, 200, 12, rnd=10)
    #bs_r = [30, 40, 50, 60, 80, 100, 150]

    #p_e_size_r = [0, 25, 50, 75]
    #p_e_size_r = [25, 50]
    #p_e_size_r = [0, 25]
    #p_e_size_r = [25, 100]
    p_e_size_r = [0] #x
    #p_e_size_r = [0]

    # Sampling
    #synl = unif_samp(range_list=synl_r, n_samps=n_iter)
    #finl = unif_samp(range_list=finl_r, n_samps=n_iter)
    synl = synl_r
    finl = finl_r

    finh = unif_samp(range_list=finh_r, n_samps=n_iter)
    #finh = [400, 500] * n_iter #x
    #finh = [400, 500, 600] * (n_iter // 2)
    frac_syn = unif_samp(range_list=frac_syn_r, n_samps=n_iter)
    #semh = unif_samp(range_list=[200, 300, 400], n_samps=n_iter)

    #lr = log_samp(e_lo=-3, e_hi=-2.5, n_samps=n_iter)
    #lr = unif_samp(range_list=[1e-3, 2e-3, 5e-4], n_samps=n_iter)
    lr = [1e-3] * n_iter #x
    #lr_sem = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_sem = unif_samp(range_list=[1, 100], n_samps=n_iter)
    #lr_sem[0] = 100
    #lr_sem = unif_samp(range_list=[1, 10, 100], n_samps=n_iter)
    #lr_sem = unif_samp(range_list=[1, 10], n_samps=n_iter)
    #lr_sem = [1] * n_iter
    lr_pos = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
    assert(len(lr_pos) == n_iter)
    #lr_sdp = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_stag = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_sem=unif_samp(range_list=scales, n_samps=n_iter)
    #lr_stag=unif_samp(range_list=scales, n_samps=n_iter)

    p_e_size = unif_samp(range_list=p_e_size_r, n_samps=n_iter)

    #sdp_bs = unif_samp(range_list=bs_r, n_samps=n_iter)
    #sdp_bs = [80] * n_iter #x
    sdp_bs = [100] * n_iter
    #sdp_bs = [100] * 6 + unif_samp(range_list=[30, 40, 80, 100, 150], n_samps=(n_iter-6)).tolist()
    #sdp_bs = unif_samp(range_list=[80, 100, 150], n_samps=n_iter)
    #sem_bs = unif_samp(range_list=[50,80,100], n_samps=n_iter)
    #sem_bs = [80] * n_iter #x
    #stag_bs = unif_samp(range_list=[20, 40, 80, 100], n_samps=n_iter)
    #stag_bs = [50, 100] * n_iter #XXX

    #ldrop = unif_samp(range_list=[0.33, 0.4, 0.5], n_samps=n_iter)
    #ldrop = unif_samp(range_list=[0.1, 0.33], n_samps=n_iter)
    ldrop = [0.33] * n_iter
    #ldrop = [0.5] * n_iter
    #edrop = unif_samp(range_list=[0, 0.25], n_samps=n_iter)
    #edrop = [0.1, 0.33] * (n_iter // 2)
    #edrop = [0.33] * n_iter
    edrop = [0.] * n_iter
    #ldrop = unif_samp(range_list=[0.5], n_samps=n_iter) #x
    #edrop = unif_samp(range_list=[0.33], n_samps=n_iter) #x

    #scramble = unif_samp(range_list=[0, 0.33], n_samps=n_iter) #x
    #scramble = [0, 0.33] * n_iter #XXX
    #scramble = [0.33] * n_iter #XXX
    scramble = [0] * n_iter
    #scramble = unif_samp(range_list=[0], n_samps=n_iter)
    #scramble = [0, 0, 0, 0.33, 0.33, 0.33] + unif_samp(range_list=[0.33], n_samps=n_iter-6).tolist()
    margin = [0.4] * n_iter
    #margin = [0.1, 0.2, 0.4, 0.6, 0.8]

    #nchunks = unif_samp(range_list=[2, 3, 5], n_samps=n_iter)
    nchunks = [1] * n_iter
    #nchunks = [1, 3] * n_iter

    #word_dropout_ = [40, 0.25, 40, 0.25, 0.25, 40] + unif_samp(range_list=[0.25, 40], n_samps=n_iter-6).tolist()
    #word_dropout = [0] * n_iter
    word_dropout = [40] * n_iter
    #word_dropout_ = unif_samp(range_list=[0.1, 0.4, 0.3, 0.2], n_samps=n_iter)
    #word_dropout_ = unif_samp(range_list=[0.05, 0.1, 0.2, 0.3], n_samps=n_iter)
    #word_dropout_ = [0.2, 0.3, 0.4] * n_iter
    #word_dropout_ = unif_samp(range_list=[0, 40], n_samps=n_iter)
    #word_dropout_ = [0] * 5 + [1] * 5 #x

    pos_dropout = [0] * n_iter

    #seed = np.random.randint(500, size=n_iter) #x
    #seed[0] = 140
    seed = [140] * n_iter
    #seed = [11, 11, 30, 30]
    #seed = [19] * n_iter 
    #seed = [21] * n_iter

    #gloved = [0] * 5 + [100] * 5
    gloved = [100] * n_iter #x
    #gloved = [100, 100, 0, 0]


    #esp = unif_samp(range_list=[2, 5], n_samps=n_iter)

    #trunc = [True] * 5 + [False] * (n_iter - 5)
    #trunc = unif_samp(range_list=[False, True], n_samps=n_iter)

    #tunk = [False, True, False, True]
    tunk = [True]

    #style = ['unif'] * n_iter
    style = ['freq'] * n_iter
    #unkstyle = ['zero'] * (n_iter // 2) + ['rand'] * (n_iter // 2)
    unkstyle = ['zero'] * n_iter
    #unkstyle = ['rand'] * n_iter

    epochs = [20] * n_iter

    #auxdrop = unif_samp(range_list=[False, True], n_samps=n_iter)
    #auxdrop = [True] * (n_iter // 2) + [False] * (n_iter // 2)
    #auxdrop = [True] * n_iter

    cuda = [0] * n_iter
    #cuda = [1] * n_iter 

    #layerdrop = [0.05, 0.3, 0.1, 0.2] * n_iter
    #layerdrop = [0] * n_iter
    #layerdrop = [0, 0.1] * (n_iter // 2)
    #layerdrop = unif_samp(range_list=[0, 0.1], n_samps=n_iter)

    #start_epoch = unif_samp(range_list=[3, 4], n_samps=n_iter)
    #start_epoch = unif_samp(range_list=[2], n_samps=n_iter)
    #scale_adv_loss = unif_samp(range_list=[1, 1e-2, 10, 1e-3], n_samps=n_iter)
    #scale_adv_loss = [3e-3] * n_iter

    for i in range(0, n_iter):
        print('\n' * 2 + f'Entering training for {i}-th model.\n')

        finl_ = finl[i]
        synl_ = synl[i]

        finh_ = finh[i]
        synh_ = 400 #XXX

        lr_ = lr[i]

        p_e_size_ = p_e_size[i]

        sdp_bs_ = sdp_bs[i]

        ldrop_ = ldrop[i]
        #edrop_ = edrop[i]

        gloved_ = gloved[i]


        seed_ = seed[i]


        style_ = style[i]

        epochs_ = epochs[i]

        cuda_ = cuda[i]

        unkstyle_ = unkstyle[i]

        model_name = f'{name_prefix[i]}_{i:02d}'
        command = f'python main.py {model_name}\
                -pe {p_e_size_}\
                -finh {finh_} -synh {synh_}\
                -finl {finl_} -synl {synl_}\
                -lr {lr_}\
                -sdpbs {sdp_bs_}\
                -epochs {epochs_}\
                -wd {word_dropout[i]}\
                -ldrop {ldrop_}\
                -seed {seed_}\
                -dropstyle {style_}\
                -unkstyle {unkstyle_}\
                -auto -cuda {cuda_}\
                -lrpos {lr_pos[i]}'

        command = command.replace('\n', '')
        command += f' -gloved {gloved_}'
        command += f' -pp ' # predict pos
        #command += f' -zw '

        #command += ' -advstag'

        #if trunc_:
        #    command += ' --trunc'

        #if tunk[i]:
        #    command += ' -tunk'

        # for resuming the semDrop trials
        #if auxdrop[i]:
        #    command += ' -wdsem'
        #    command += ' -wdstag'

        subprocess.call(command + ' -tm 0', shell=True) #x
        #subprocess.call(command + ' -tm 1', shell=True)
        #subprocess.call(command + ' -tm 2', shell=True)
        #subprocess.call(command + ' -tm 0 1 2', shell=True)
        #subprocess.call(command + ' -estag', shell=True) #x
        #subprocess.call(command + ' -esem', shell=True) #x
        #subprocess.call(command + ' -e 0', shell=True)
        subprocess.call(command + ' -e 0 2', shell=True) #x



def unif_samp(range_list=None, n_samps=None):
    sample = np.random.multinomial(1, [1 / len(range_list)] * len(range_list), size=n_samps)
    sample = np.argmax(sample, axis=1)

    return np.array(range_list).take(sample)


#def log_samp(exponent=None, n_samps=None):
def log_samp(e_lo=None, e_hi=0, n_samps=None):
    # e <-> exponent
    if e_lo >= 0 or e_hi > 0 or e_lo >= e_hi:
        raise Exception
    sample = ((e_lo - e_hi) * np.random.rand(n_samps)) + e_hi

    return 10 ** sample


def logrange_list(low, high, pts, rnd=None):
    a = np.logspace(np.log10(low), np.log10(high), pts)
    a = rnd * np.round(a / rnd)
    a = np.unique(a)

    return a


# Needed so that main function has access to functions in this file
if __name__ == '__main__':
    main()

