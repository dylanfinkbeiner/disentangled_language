import subprocess
import math
import sys

import numpy as np


def main():
    #name_prefix = input('Prefix for model names: ').strip()

    #n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_iter =  10 #x
    #n_iter =  50

    #name_prefix = ['semDrop'] * 6 + ['vanillaTuning'] * (n_iter - 6)
    #name_prefix = ['mimi23trials'] * n_iter
    name_prefix = ['OURBEST'] * n_iter #x
    #name_prefix = ['2semDrop'] * n_iter

    # Ranges
    #seml_r = [1]
    #synl_r = [1, 2]
    #synl_r = [2, 3]
    synl_r = [1] # x
    #synl_r = [2] 
    #finl_r = [1, 2, 3]
    #finl_r = [1, 2]
    finl_r = [2] #x
    #finl_r = [1] 

    #finh_r = [300, 400, 500, 600]
    #finh_r = [400, 500]
    frac_syn_r = [1/8 * x for x in range(4, 6)]

    #bs_r = logrange_list(20, 200, 12, rnd=10)
    bs_r = [30, 40, 50, 60, 80, 100, 150]

    #p_e_size_r = [0, 25, 50, 75]
    #p_e_size_r = [25, 50]
    #p_e_size_r = [0, 25]
    #p_e_size_r = [25, 100]
    p_e_size_r = [25] #x

    # Sampling
    synl = unif_samp(range_list=synl_r, n_samps=n_iter)
    finl = unif_samp(range_list=finl_r, n_samps=n_iter)

    #finh = unif_samp(range_list=finh_r, n_samps=n_iter)
    finh = [400] * n_iter #x
    #finh = [400, 500] * (n_iter // 2)
    frac_syn = unif_samp(range_list=frac_syn_r, n_samps=n_iter)
    #semh = unif_samp(range_list=[200, 300, 400], n_samps=n_iter)

    #lr = log_samp(e_lo=-3, e_hi=-2.5, n_samps=n_iter)
    #lr = unif_samp(range_list=[1e-3, 2e-3, 5e-4], n_samps=n_iter)
    lr = [1e-3] * n_iter #x
    #lr_sem = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_sem = unif_samp(range_list=[1, 10, 100], n_samps=n_iter)
    lr_sem = unif_samp(range_list=[1], n_samps=n_iter) #x
    #lr_sdp = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_stag = log_samp(e_lo=-2, n_samps=n_iter)
    #lr_sem=unif_samp(range_list=scales, n_samps=n_iter)
    #lr_stag=unif_samp(range_list=scales, n_samps=n_iter)

    p_e_size = unif_samp(range_list=p_e_size_r, n_samps=n_iter)

    #sdp_bs = unif_samp(range_list=bs_r, n_samps=n_iter)
    sdp_bs = [80] * n_iter #x
    #sdp_bs = [100] * n_iter
    #sdp_bs = [100] * 6 + unif_samp(range_list=[30, 40, 80, 100, 150], n_samps=(n_iter-6)).tolist()
    #sdp_bs = unif_samp(range_list=[80, 100, 150], n_samps=n_iter)
    #sem_bs = unif_samp(range_list=[50,80,100], n_samps=n_iter)
    sem_bs = [80] * n_iter #x
    #sem_bs = [100] * n_iter
    #stag_bs = unif_samp(range_list=[20, 40, 80, 100], n_samps=n_iter)
    stag_bs = [50] * n_iter #x

    #ldrop = unif_samp(range_list=[0.33, 0.4, 0.5], n_samps=n_iter)
    #edrop = unif_samp(range_list=[0.33, 0.4, 0.5], n_samps=n_iter)
    #ldrop = unif_samp(range_list=[0, 0.15, 0.33], n_samps=n_iter)
    #edrop = unif_samp(range_list=[0, 0.15, 0.33], n_samps=n_iter)
    ldrop = unif_samp(range_list=[0.5], n_samps=n_iter) #x
    edrop = unif_samp(range_list=[0.33], n_samps=n_iter) #x

    scramble = unif_samp(range_list=[0.33], n_samps=n_iter) #x
    #scramble = unif_samp(range_list=[0], n_samps=n_iter)
    #scramble = [0, 0, 0, 0.33, 0.33, 0.33] + unif_samp(range_list=[0.33], n_samps=n_iter-6).tolist()
    margin = unif_samp(range_list=[0.4], n_samps=n_iter) #x

    #nchunks = unif_samp(range_list=[2, 3, 5], n_samps=n_iter)
    nchunks = unif_samp(range_list=[1], n_samps=n_iter) #x

    #alpha = [40, 0.25, 40, 0.25, 0.25, 40] + unif_samp(range_list=[0.25, 40], n_samps=n_iter-6).tolist()
    #alpha = [0] * n_iter
    #alpha = unif_samp(range_list=[0, 1, 40], n_samps=n_iter)
    alpha = [0] * 5 + [1] * 5 #x

    seed = np.random.randint(500, size=n_iter) #x
    #seed = [1] * n_iter

    #gloved = [0] * 5 + [100] * 5
    gloved = [100] * n_iter #x


    #esp = unif_samp(range_list=[2, 5], n_samps=n_iter)

    #trunc = [True] * 5 + [False] * (n_iter - 5)


    for i in range(0, 4):
        print('\n' * 2 + f'Entering training for {i}-th model.\n')

        finl_ = finl[i]
        synl_ = synl[i]
        #synl_ = 1 if finl_ == 2 else synl[i]
        #if synl_ == 1 and finl_ == 1:
        #    synl_ = 2

        finh_ = finh[i]
        #synh_ = finh[i]
        #synh_ = 25 * math.floor((finh_ * frac_syn[i]) / 25)
        synh_ = 200 #x
        semh_ = finh_ - synh_
        #semh_ = semh[i]
        #semh_ = 0

        lr_ = lr[i]
        #lr_sdp_ = lr_sdp[i]
        lr_sem_ = lr_sem[i]
        #lr_stag_ = lr_stag[i]
        lr_sdp_ = 1
        #lr_sem_ = 1
        lr_stag_ = 1

        p_e_size_ = p_e_size[i]

        sem_bs_ = sem_bs[i]
        sdp_bs_ = sdp_bs[i]
        stag_bs_ = stag_bs[i]

        ldrop_ = ldrop[i]
        edrop_ = edrop[i]

        scramble_ = scramble[i]

        alpha_ = alpha[i]

        gloved_ = gloved[i]

        margin_ = margin[i]

        nchunks_ = nchunks[i]

        seed_ = seed[i]

        #esp_ = esp[i]

        #trunc_ = trunc[i]

        model_name = f'{name_prefix[i]}_{i:02d}'
        command = f'python main.py {model_name}\
                -pe {p_e_size_}\
                -finh {finh_} -synh {synh_} -semh {semh_}\
                -finl {finl_} -synl {synl_} -seml 1\
                -lr {lr_} -lrsdp {lr_sdp_} -lrstag {lr_stag_} -lrsem {lr_sem_}\
                -sdpbs {sdp_bs_} -stagbs {stag_bs_} -sembs {sem_bs_}\
                --nchunks {nchunks_} -M 20 --epochs 25 --gloved {gloved_} --scramble {scramble_} -a {alpha_}\
                --margin {margin_}\
                -ldrop {ldrop_} -edrop {edrop_}\
                --seed {seed_}\
                -auto -cuda 0' #x
                #-auto -cuda 1'

        command = command.replace('\n', '')

        #command += ' -sdrop'

        # for resuming the semDrop trials
        #command += ' -wdsem'
        #command += ' -wdstag'

        #subprocess.call(command + ' -tm 2', shell=True)
        #subprocess.call(command + ' -tm 1', shell=True)
        subprocess.call(command + ' -tm 0 1 2', shell=True) #x
        subprocess.call(command + ' -estag', shell=True) #x
        subprocess.call(command + ' -esem', shell=True) #x
        #subprocess.call(command + ' -tm 0', shell=True)
        subprocess.call(command + ' -e 0 2', shell=True) #x
        #subprocess.call(command + ' -e 0', shell=True)

        #subprocess.call(command + ' -tm 1', shell=True)
        #subprocess.call(command + ' -esem', shell=True)


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

