__author__ = 'yihanjiang'

from utils import build_rnn_data_feed

from turbo_rnn import load_model

import sys
import numpy as np
import time

import logging

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv

if __name__ == '__main__':
    ##########################################
    # Loading Arguments
    ##########################################
    n_inp = sys.argv[1:]

    if '--help' in n_inp:
        with open('./manuals/evaluate_rnn.md','r') as fin:
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
        num_block = 10

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
    denoise_thd   = 10.0

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

    if '-network_model_path' in n_inp:
        ind1      = n_inp.index('-network_model_path')
        model_path = str(n_inp[ind1+1])
    else:
        model_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    print '[Setting Parameters] Network Saved Path is ', model_path

    if '-codec_type' in n_inp:
        ind1      = n_inp.index('-codec_type')
        codec_type = str(n_inp[ind1+1])
    else:
        codec_type = 'default'

    print '[Setting Parameters]Codec type is ', codec_type

    if '-num_hidden_unit' in n_inp:
        ind1      = n_inp.index('-num_hidden_unit')
        num_hidden_unit = int(n_inp[ind1+1])
    else:
        num_hidden_unit = 200

    print '[Setting Parameters]RNN Number of hidden unit ', num_hidden_unit

    if '-rnn_type' in n_inp:
        ind1      = n_inp.index('-rnn_type')
        rnn_type = str(n_inp[ind1+1])
    else:
        rnn_type = 'lstm'

    print '[Setting Parameters]RNN type is  ', rnn_type

    if '-rnn_direction' in n_inp:
        ind1      = n_inp.index('-rnn_direction')
        rnn_direction = str(n_inp[ind1+1])
    else:
        rnn_direction = 'bd'

    print '[Setting Parameters]RNN direction is  ', rnn_direction


    if '-num_layer' in n_inp:
        ind1      = n_inp.index('-num_layer')
        num_layer = int(n_inp[ind1+1])
    else:
        num_layer = 2

    print '[Setting Parameters]RNN number of layer is  ', num_layer

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
    # Systematic Bit Channel Reliability
    ##########################################
    if '--add_sys' in n_inp:
        add_sys = True
        last_layer_sigmoid = False
    else:
        add_sys = False
        last_layer_sigmoid = True

    print '[helper] add sys ', add_sys, 'last layer sigmoid ', last_layer_sigmoid

    ##########################################
    # Setting Up RNN Model
    ##########################################
    start_time = time.time()

    model = load_model(network_saved_path = model_path, block_len=block_len,rnn_type=rnn_type, num_layer= num_layer,
                       rnn_direction = rnn_direction,
                       last_layer_sigmoid = last_layer_sigmoid,
                       interleave_array = p_array, dec_iter_num = dec_iter_num, num_hidden_unit=num_hidden_unit)
    end_time = time.time()
    print '[RNN decoder]loading RNN model takes ', str(end_time-start_time), ' secs'   # typically longer than 5 mins, since it is deep!

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

    turbo_res_ber = []
    turbo_res_bler= []

    for idx in xrange(SNR_points):
        start_time = time.time()
        noiser = [noise_type, test_sigmas[idx], vv, radar_power, radar_prob, denoise_thd]

        X_feed_test, X_message_test = build_rnn_data_feed(num_block, block_len, noiser, codec)
        pd       = model.predict(X_feed_test,batch_size = 100)

        if add_sys:
            weighted_sys = 2*X_feed_test[:,:,0]*1.0/(test_sigmas[idx]**2)
            weighted_sys = weighted_sys.reshape((weighted_sys.shape[0], weighted_sys.shape[1], 1))
            if last_layer_sigmoid == False:
                decoded_bits = (pd + weighted_sys > 0)
            else:
                print 'not supported, halt!'
                sys.exit()
        else:
            decoded_bits = np.round(pd)


        # Compute BER and BLER
        ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_message_test))))*1.0/(X_message_test.shape[0]*X_message_test.shape[1])# model.evaluate(X_feed_test, X_message_test, batch_size=10)
        tp0 = (abs(decoded_bits-X_message_test)).reshape([X_message_test.shape[0],X_message_test.shape[1]])
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_message_test.shape[0])

        print '[testing] This is SNR', SNRS[idx] , 'RNN BER ', ber_err_rate, 'RNN BLER', bler_err_rate
        turbo_res_ber.append(ber_err_rate)
        turbo_res_bler.append(bler_err_rate)
        end_time = time.time()
        print '[testing] runnig time is', str(end_time-start_time)

    print '[Result Summary] SNRS is', SNRS
    print '[Result Summary] Turbo RNN BER is', turbo_res_ber
    print '[Result Summary] Turbo RNN BLER is', turbo_res_bler