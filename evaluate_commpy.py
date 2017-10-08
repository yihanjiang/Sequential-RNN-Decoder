__author__ = 'yihanjiang'
'''
This module is for future speedup of Commpy.
'''

from utils import  corrupt_signal, snr_db2sigma

import sys
import numpy as np
import time

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv

from commpy.utilities import hamming_dist
import commpy.channelcoding.turbo as turbo

import multiprocessing as mp

if __name__ == '__main__':
    ##########################################
    # Loading Arguments
    ##########################################
    n_inp = sys.argv[1:]

    if '--help' in n_inp:
        with open('./manuals/evaluate_commpy.md','r') as fin:
            print fin.read()
        exit_now = True
        sys.exit()
    print '[Turbo Decoder] Evaluate Commpy Turbo Decoder only, no RNN!'
    if '-block_len' in n_inp:
        ind1      = n_inp.index('-block_len')
        block_len = int(n_inp[ind1+1])
    else:
        block_len = 1000
    print '[Setting Parameters] Code Block Length is ', block_len

    if '-num_block' in n_inp:
        ind1      = n_inp.index('-num_block')
        num_block = int(n_inp[ind1+1])
    else:
        num_block = 100

    print '[Setting Parameters] Number of Block is ', num_block

    if '-num_dec_iteration' in n_inp:
        ind1      = n_inp.index('-num_dec_iteration')
        dec_iter_num = int(n_inp[ind1+1])
    else:
        dec_iter_num = 6

    print '[Setting Parameters] Turbo Decoding Iteration is ', dec_iter_num

    if '-num_cpu' in n_inp:
        ind1      = n_inp.index('-num_cpu')
        num_cpu = int(n_inp[ind1+1])
    else:
        num_cpu = 5

    print '[Setting Parameters] Number of Processes is ', num_cpu

    if '-noise_type' in n_inp:
        ind1      = n_inp.index('-noise_type')
        noise_type = str(n_inp[ind1+1])
    else:
        noise_type = 'awgn'

    vv          = -1.0
    radar_power = -1.0
    radar_prob  = -1.0
    denoise_thd = 10.0

    if noise_type == 'awgn':
        print '[Setting Parameters] Noise Type is ', noise_type

    elif noise_type == 't-dist':
        if '-v' in n_inp:
            ind1 = n_inp.index('-v')
            vv   = float(n_inp[ind1+1])
        else:
            vv   = 5.0
        print '[Setting Parameters] Noise Type is ', noise_type, ' with v=', vv

    elif noise_type == 'awgn+radar' or noise_type == 'hyeji_bursty':
        if '-radar_power' in n_inp:
            ind1 = n_inp.index('-radar_power')
            radar_power   = float(n_inp[ind1+1])
        else:
            radar_power   = 20.0

        if '-radar_prob' in n_inp:
            ind1 = n_inp.index('-radar_prob')
            radar_prob   = float(n_inp[ind1+1])
        else:
            radar_prob   = 5e-2

        print '[Setting Parameters] Noise Type is ', noise_type, 'with Radar Power ', radar_power, ' with Radar Probability ', radar_prob

    elif noise_type == 'awgn+radar+denoise' or noise_type == 'hyeji_bursty+denoise':
        if '-radar_power' in n_inp:
            ind1 = n_inp.index('-radar_power')
            radar_power   = float(n_inp[ind1+1])
        else:
            radar_power   = 20.0

        if '-radar_prob' in n_inp:
            ind1 = n_inp.index('-radar_prob')
            radar_prob   = float(n_inp[ind1+1])
        else:
            radar_prob   = 5e-2

        if '-denoise_thd' in n_inp:
            ind1 = n_inp.index('-denoise_thd')
            denoise_thd   = float(n_inp[ind1+1])
        else:
            denoise_thd   = 10.0

        print '[Setting Parameters] Using Thresholding Denoise with denoise threshold', denoise_thd
        print '[Setting Parameters] Noise Type is ', noise_type, 'with Radar Power ', radar_power, ' with Radar Probability ', radar_prob

    if '-snr_range' in n_inp:
        ind1      = n_inp.index('-snr_range')
        snr_start = float(n_inp[ind1+1])
        snr_stop  = float(n_inp[ind1+2])
    else:
        snr_start = -1.5
        snr_stop  = 2

    print '[Setting Parameters] SNR in dB range from ', snr_start,' to ', snr_stop, 'dB'

    if '-snr_points' in n_inp:
        ind1      = n_inp.index('-snr_points')
        snr_points = int(n_inp[ind1+1])
    else:
        snr_points = 8

    print '[Setting Parameters] SNR points in total: ', snr_points

    if '-fix_var' in n_inp:
        is_fixed_var = True
        ind1      = n_inp.index('-fix_var')
        fix_var_db = float(n_inp[ind1+1])
    else:
        is_fixed_var = False
        fix_var_db = -1.0

    fix_var = snr_db2sigma(fix_var_db)
    print '[Setting Parameters] setting fixed variance', is_fixed_var, 'with variance in dB', fix_var_db, 'with variance', fix_var

    if '-codec_type' in n_inp:
        ind1      = n_inp.index('-codec_type')
        codec_type = str(n_inp[ind1+1])
    else:
        codec_type = 'default'

    print '[Setting Parameters]Codec type is ', codec_type

    ##########################################
    # Setting Up Codec
    ##########################################
    if codec_type == 'lte':
        M = np.array([3]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[11,13]])
        feedback = 11
    else:     #'defalut'
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])
        feedback = 7

    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(block_len, 0)
    p_array = interleaver.p_array
    print '[BCJR Example Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback
    codec  = [trellis1, trellis2, interleaver]



    ##########################################
    # Setting Up Channel & SNR range
    ##########################################
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(float(num_block)/float(num_block*2.0)) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print '[testing] SNR range in dB ', SNRS

    tic = time.time()

    def turbo_compute((idx, x)):
        '''
        Compute Turbo Decoding in 1 iterations for one SNR point.
        '''
        np.random.seed()
        message_bits = np.random.randint(0, 2, block_len)
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        #print 'noiser', noise_type, vv, radar_power, radar_prob, denoise_thd

        sys_r  = corrupt_signal(sys, noise_type =noise_type, sigma = test_sigmas[idx],
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)
        par1_r = corrupt_signal(par1, noise_type =noise_type, sigma = test_sigmas[idx],
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)
        par2_r = corrupt_signal(par2, noise_type =noise_type, sigma = test_sigmas[idx],
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)

        #decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, test_sigmas[idx]**2, dec_iter_num, interleaver, L_int = None)
        if is_fixed_var:
            decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, fix_var**2, dec_iter_num, interleaver, L_int = None)
        else:
            decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, test_sigmas[idx]**2, dec_iter_num, interleaver, L_int = None)

        num_bit_errors = hamming_dist(message_bits, decoded_bits)

        return num_bit_errors


    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors = np.zeros(test_sigmas.shape)
    map_nb_errors = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        start_time = time.time()

        pool = mp.Pool(processes=num_cpu)
        results = pool.map(turbo_compute, [(idx,x) for x in range(num_block)])

        for result in results:
            if result == 0:
                nb_block_no_errors[idx] = nb_block_no_errors[idx]+1

        nb_errors[idx]+= sum(results)
        print '[testing]SNR: ' , SNRS[idx]
        print '[testing]BER: ', sum(results)/float(block_len*num_block)
        print '[testing]BLER: ', 1.0 - nb_block_no_errors[idx]/num_block
        commpy_res_ber.append(sum(results)/float(block_len*num_block))
        commpy_res_bler.append(1.0 - nb_block_no_errors[idx]/num_block)
        end_time = time.time()
        print '[testing] This SNR runnig time is', str(end_time-start_time)


    print '[Result]SNR: ', SNRS
    print '[Result]BER', commpy_res_ber
    print '[Result]BLER', commpy_res_bler

    toc = time.time()

    print '[Result]Total Running time:', toc-tic
