from turbo_rnn import load_model
from utils import build_rnn_data_feed, get_test_sigmas
import tensorflow as tf
import sys
import numpy as np
import time
import logging
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_dec_iteration', type=int, default=6)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")


    parser.add_argument('-snr_test_start', type=float, default=-1.5)
    parser.add_argument('-snr_test_end', type=float, default=2.0)
    parser.add_argument('-snr_points', type=int, default=8)

    parser.add_argument('-model_path', type=str, default='./models/turbo_models/awgn_bl100_1014.h5')

    parser.add_argument('-rnn_type', choices = ['lstm', 'gru'], default = 'lstm')
    parser.add_argument('-rnn_direction', choices = ['bd', 'sd'], default = 'bd')
    parser.add_argument('-num_layer', type=int, default=2)
    parser.add_argument('-num_hidden_unit', type=int, default=200)

    parser.add_argument('-batch_size',  type=int, default=10)

    parser.add_argument('-noise_type', choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-radar_power', type=float, default=20.0)
    parser.add_argument('-radar_prob', type=float, default=0.05)

    parser.add_argument('-fixed_var', type=float, default=0.00)
    parser.add_argument('--GPU_proportion', type=float, default=1.00)
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()
    print args
    print '[ID]', args.id
    return args


if __name__ == '__main__':
    ##################################################################################################################
    # Parse Arguments
    ##################################################################################################################
    args = get_args()

    if args.GPU_proportion < 1.00:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        frac = args.GPU_proportion

        config.gpu_options.per_process_gpu_memory_fraction = frac
        set_session(tf.Session(config=config))
        print '[Test][Warining] Restrict GPU memory usage to 45%, enable',str(int(1/args.GPU_proportion)), 'processes'

    print '[Setting Parameters] Number of Block is ', args.num_block

    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    p_array = interleaver.p_array
    print '[Convolutional Code Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback
    codec  = [trellis1, trellis2, interleaver]

    ##########################################
    # Setting Up RNN Model
    ##########################################
    start_time = time.time()

    model = load_model(network_saved_path = args.model_path, block_len=args.block_len,
                       rnn_type=args.rnn_type, num_layer= args.num_layer,
                       rnn_direction = args.rnn_direction,
                       interleave_array = p_array, dec_iter_num = args.num_dec_iteration, num_hidden_unit=args.num_hidden_unit)
    end_time = time.time()
    print '[RNN decoder]loading RNN model takes ', str(end_time-start_time), ' secs'   # typically longer than 5 mins, since it is deep!

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    turbo_res_ber = []
    turbo_res_bler= []

    for idx in xrange(len(test_sigmas)):
        start_time = time.time()
        noiser = [args.noise_type, test_sigmas[idx]]
        X_feed_test, X_message_test = build_rnn_data_feed(args.num_block, args.block_len, noiser, codec)
        pd       = model.predict(X_feed_test)
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