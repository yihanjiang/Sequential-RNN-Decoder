__author__ = 'yihanjiang'

from turbo_rnn import load_model
from utils import build_rnn_data_feed

import sys
import numpy as np
import time

import logging

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
frac = 0.45

config.gpu_options.per_process_gpu_memory_fraction = frac
set_session(tf.Session(config=config))

print '[Test][Warining] Restrict GPU memory usage to 45%, enable',str(int(1/0.45)), 'processes'


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_dec_iteration', type=int, default=6)


    parser.add_argument('-code_rate',  type=int, default=3)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)

    parser.add_argument('-num_cpu', type=int, default=1)

    parser.add_argument('-snr_test_start', type=float, default=-1.0)
    parser.add_argument('-snr_test_end', type=float, default=8.0)
    parser.add_argument('-snr_points', type=int, default=10)

    parser.add_argument('-noise_type', choices = ['awgn', 't-dist','hyeji_bursty' ], default='awgn')
    parser.add_argument('-radar_power', type=float, default=20.0)
    parser.add_argument('-radar_prob', type=float, default=0.05)

    parser.add_argument('-fixed_var', type=float, default=0.00)

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    num_layer
    parser.add_argument('-num_layer', type=int, default=2)
    parser.add_argument('-num_unit', type=int, default=200)
    parser.add_argument('-rnn_type', choices=['lstm', 'gru', 'simplernn'], default='lstm')
    parser.add_argument('-rnn_direction', choices=['bd', 'sd'], default='bd')
    parser.add_argument('-model_path', type=str, default='./model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5')

    args = parser.parse_args()

    print '[ID]', args.id
    return args


def get_nn_param(args):
    model_weight_path = args.model_path
    num_unit   = args.num_unit




if __name__ == '__main__':

    args = get_args()


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
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
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
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(float(num_block)/float(num_block*2.0)) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print '[testing] SNR range in dB ', SNRS

    turbo_res_ber = []
    turbo_res_bler= []

    for idx in xrange(SNR_points):
        start_time = time.time()
        noiser = [noise_type, test_sigmas[idx], vv, radar_power, radar_prob, denoise_thd, snr_mix]

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