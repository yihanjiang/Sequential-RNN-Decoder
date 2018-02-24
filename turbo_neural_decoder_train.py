
from utils import build_rnn_data_feed
from turbo_rnn import load_model

import sys
import numpy as np
import time

import keras
import tensorflow as tf

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_block_train', type=int, default=100)
    parser.add_argument('-num_block_test', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_dec_iteration', type=int, default=6)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")

    parser.add_argument('-init_nw_model', type=str, default='./models/turbo_models/awgn_bl100_1014.h5')

    parser.add_argument('-rnn_type', choices = ['lstm', 'gru'], default = 'lstm')
    parser.add_argument('-rnn_direction', choices = ['bd', 'sd'], default = 'bd')
    parser.add_argument('-num_layer', type=int, default=2)
    parser.add_argument('-num_hidden_unit', type=int, default=200)

    parser.add_argument('-batch_size',  type=int, default=10)
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-noise_type', choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-train_snr', type=float, default=-1.0)
    parser.add_argument('-train_loss', choices = ['binary_crossentropy', 'mse', 'mae'], default='binary_crossentropy')

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
    args = get_args()

    if args.GPU_proportion < 1.00:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        frac = args.GPU_proportion

        config.gpu_options.per_process_gpu_memory_fraction = frac
        set_session(tf.Session(config=config))
        print '[Test][Warining] Restrict GPU memory usage to 45%, enable',str(int(1/args.GPU_proportion)), 'processes'


    print '[Setting Parameters] Number of Train Block is ',    args.num_block_train, ' Test Block ', args.num_block_test
    print '[Setting Parameters] Turbo Decoding Iteration is ', args.num_dec_iteration
    print '[Setting Parameters] Noise Type is ',               args.noise_type
    print '[Setting Parameters] Training Data SNR is ',        args.train_snr, ' dB'
    print '[Setting Parameters] Training Loss is ',            args.train_loss
    print '[Setting Parameters] Network starting path is ',    args.init_nw_model
    print '[Setting Parameters]Trained Model Weight saving path is at: ', './tmp'+args.id
    print '[Setting Parameters]Initial learning_rate is ',     args.learning_rate
    print '[Setting Parameters]Training batch_size is ',       args.batch_size
    print '[Setting Parameters]Training num_epoch is ',        args.num_epoch
    print '[Setting Parameters]RNN Number of hidden unit ',    args.num_hidden_unit
    print '[Setting Parameters]RNN type is  ',                 args.rnn_type


    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    p_array = interleaver.p_array
    print '[Convolutional Code Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback
    codec  = [trellis1, trellis2, interleaver]


    start_time = time.time()

    model = load_model(learning_rate=args.learning_rate,rnn_type=args.rnn_type, block_len=args.block_len,
                       network_saved_path = args.init_nw_model, num_hidden_unit=args.num_hidden_unit,
                       interleave_array = p_array, dec_iter_num = args.num_dec_iteration, loss=args.train_loss)

    end_time = time.time()
    print '[RNN decoder]loading RNN model takes ', str(end_time-start_time), ' secs'

    ##########################################
    # Setting Up Channel & Train SNR
    ##########################################
    train_snr_Es = args.train_snr + 10*np.log10(float(args.block_len)/float(2*args.block_len))
    sigma_snr  = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    SNR = -10*np.log10(sigma_snr**2)

    noiser = [args.noise_type, sigma_snr]  # For now only AWGN is supported
    start_time = time.time()

    X_feed_test, X_message_test = build_rnn_data_feed(args.num_block_test,  args.block_len, noiser, codec)
    X_feed_train,X_message_train= build_rnn_data_feed(args.num_block_train, args.block_len, noiser, codec)

    save_path = './tmp/weights_' + args.id+ '.h5'
    model.save_weights(save_path)
    print '[Warning] Save every epoch', './tmp/weights_' + args.id+ '.h5'


    save_cb = keras.callbacks.ModelCheckpoint('./tmp/save'+args.id+ '_{epoch:02d}-{val_loss:.2f}' +'.h5', monitor='val_loss', verbose=0,
                                              save_best_only=False, save_weights_only=True, mode='auto', period=1)


    model.fit(x=X_feed_train, y=X_message_train, batch_size=args.batch_size,
              epochs=args.num_epoch, validation_data=(X_feed_test, X_message_test))  # starts training


    print '[Training] saved model in ', save_path

    print '[Training]This is SNR', SNR ,'Training'
    pd       = model.predict(X_feed_test,batch_size = 100)
    err_rate = sum(sum(sum(abs(np.round(pd)-X_message_test))))*1.0/(X_message_test.shape[0]*X_message_test.shape[1])

    print model.evaluate(X_feed_test, X_message_test, batch_size=10)
    print '[Training]RNN turbo decoding has error rate ', err_rate
    end_time = time.time()
    print '[Trainiing]Training time is', str(end_time-start_time)


