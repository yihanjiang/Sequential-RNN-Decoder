__author__ = 'yihanjiang'
'''
This module is for future speedup of Commpy.
'''

from utils import build_rnn_data_feed

import sys
import numpy as np
import time
from scipy import stats

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv

from commpy.utilities import hamming_dist
import commpy.channelcoding.turbo as turbo

if __name__ == '__main__':
    ##########################################
    # Loading Arguments
    ##########################################
    n_inp = sys.argv[1:]
    print '[Turbo Decoder] Evaluate Commpy Turbo Decoder only, no RNN!'
    if '--help' in n_inp:
        with open('evaluate_rnn.md','r') as fin:
            print fin.read()
        exit_now = True
        sys.exit()

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

    elif noise_type == 'awgn+radar':
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

    elif noise_type == 'awgn+radar+denoise':
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
        snr_start = int(n_inp[ind1+1])
        snr_stop  = int(n_inp[ind1+2])
    else:
        snr_start = -1
        snr_stop  = 2

    print '[Setting Parameters] SNR in dB range from ', snr_start,' to ', snr_stop, 'dB'

    if '-snr_points' in n_inp:
        ind1      = n_inp.index('-snr_points')
        snr_points = int(n_inp[ind1+1])
    else:
        snr_points = 10

    print '[Setting Parameters] SNR points in total: ', snr_points

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
    SNR_dB_stop_Eb  = snr_stop
    SNR_points      = snr_points

    SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(float(block_len)/float(2*block_len))
    SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(float(block_len)/float(2*block_len))

    sigma_start = np.sqrt(1/(2*10**(float(SNR_dB_start_Es)/float(10))))
    sigma_stop = np.sqrt(1/(2*10**(float(SNR_dB_stop_Es)/float(10))))

    # Testing sigmas for Direct link, Tx to Relay, Relay to Rx.
    test_sigmas = np.linspace(sigma_start, sigma_stop, SNR_points, dtype = 'float32')
    SNRS = -10*np.log10(test_sigmas**2)
    print '[testing] SNR range in dB ', SNRS

    commpy_res = []

    for idx in xrange(SNR_points):
        start_time = time.time()
        noiser = [noise_type, test_sigmas[idx], vv, radar_power, radar_prob, denoise_thd]
        X_feed_test, X_message_test = build_rnn_data_feed(num_block, block_len, noiser, codec)
        nb_errors = np.zeros([num_block,1])

        for block_idx in range(num_block):
            message_bits =  X_message_test[block_idx, :, :].T[0]
            encoded      =  X_feed_test[block_idx, :, :]
            sys_r        =  encoded[:, 0]
            par1_r       =  encoded[:, 1]
            par2_r       =  encoded[:, 4]

            decoded_bits = turbo.turbo_decode(sys_r, par1_r, par2_r, trellis1, test_sigmas[idx]**2,
                                              dec_iter_num, interleaver, L_int = None)

            # print decoded_bits
            # print message_bits
            # print haha

            num_bit_errors = hamming_dist(message_bits, decoded_bits)
            nb_errors[block_idx] = num_bit_errors

        print '[testing] This is SNR', SNRS[idx] , 'Commpy BER', float(sum(nb_errors)*1.0/(num_block*block_len))

        commpy_res.append(float(sum(nb_errors)*1.0/(num_block*block_len)) )

        end_time = time.time()
        print '[testing] runnig time is', str(end_time-start_time)


    print '[Result Summary] SNRS is', SNRS
    print '[Result Summary] Turbo Commpy BER is', commpy_res
