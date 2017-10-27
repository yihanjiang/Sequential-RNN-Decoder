__author__ = 'yihanjiang'

from utils import build_rnn_data_feed
from turbo_rnn import load_model

import sys
import numpy as np
import time

import keras

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv

if __name__ == '__main__':
    ##########################################
    # Loading Arguments
    ##########################################
    print 'TBDs: (1) logging is not done yet!'



    n_inp = sys.argv[1:]

    if '--help' in n_inp:
        with open('manual.md','r') as fin:
            print fin.read()
        exit_now = True
        sys.exit()

    if '--share_gpu' in n_inp:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.48
        set_session(tf.Session(config=config))

        print '[Test][Warining] Restrict GPU memory usage to 48%'

    if '--save_every_epoch' in n_inp:
        import keras
        save_epoch_cb = keras.callbacks.ModelCheckpoint('./tmp/', monitor='val_loss', verbose=0,
                                                        save_best_only=False, save_weights_only=True, mode='auto',
                                                        period=1)

    if '-block_len' in n_inp:
        ind1      = n_inp.index('-block_len')
        block_len = int(n_inp[ind1+1])
    else:
        block_len = 100
    print '[Setting Parameters] Code Block Length is ', block_len

    if '-num_block_train' in n_inp:
        ind1            = n_inp.index('-num_block_train')
        num_block_train = int(n_inp[ind1+1])
    else:
        num_block_train = 100

    if '-num_block_test' in n_inp:
        ind1           = n_inp.index('-num_block_test')
        num_block_test = int(n_inp[ind1+1])
    else:
        num_block_test = 10

    print '[Setting Parameters] Number of Train Block is ', num_block_train, ' Test Block ', num_block_test

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
    snr_mix     = []

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

    elif noise_type == 'mix_snr_turbo':
        if '-mix_snr' in n_inp:
            ind1 = n_inp.index('-mix_snr')
            snr_0 = float(n_inp[ind1+1])
            snr_1  = float(n_inp[ind1+2])
            snr_2  = float(n_inp[ind1+3])
        else:
            snr_0  = 0.0
            snr_1  = 2.0
            snr_2  = 4.0
        snr_mix = [snr_0, snr_1, snr_2]

        print '[Warning] mixing snr in different bit with snr 0, 1, 2', snr_mix
        from utils import snr_db2sigma
        snr_mix = [snr_db2sigma(item) for item in snr_mix]

    elif noise_type == 'mix_snr_turbo' or noise_type == 'random_snr_turbo':
        if '-mix_snr' in n_inp:
            ind1 = n_inp.index('-mix_snr')
            snr_0 = float(n_inp[ind1+1])
            snr_1  = float(n_inp[ind1+2])
            snr_2  = float(n_inp[ind1+3])
        else:
            snr_0  = -1.0
            snr_1  = 0.0
            snr_2  = 1.0
        snr_mix = [snr_0, snr_1, snr_2]

        print '[Warning] mixing snr in different bit with snr 0, 1, 2', snr_mix, 'converting from dB to sigma'
        from utils import snr_db2sigma
        snr_mix = [snr_db2sigma(item) for item in snr_mix]


    if '-train_snr' in n_inp:
        ind1      = n_inp.index('-train_snr')
        train_snr = float(n_inp[ind1+1])
    else:
        train_snr = -1.0

    print '[Setting Parameters] Training Data SNR is ', train_snr, ' dB'

    if '-train_loss' in n_inp:
        ind1      = n_inp.index('-train_loss')
        train_loss = str(n_inp[ind1+1])
    else:
        train_loss = 'binary_crossentropy'
    print '[Setting Parameters] Training Loss is ', train_loss

    if '-network_model_path' in n_inp:
        ind1      = n_inp.index('-network_model_path')
        starting_model_path = str(n_inp[ind1+1])
    else:
        starting_model_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    print '[Setting Parameters] Network starting path is ', starting_model_path

    if '-model_save_path' in n_inp:
        ind1      = n_inp.index('-model_save_path')
        model_save_path = str(n_inp[ind1+1])
    else:
        model_save_path = './tmp/'

    print '[Setting Parameters]Trained Model Weight saving path is ', model_save_path

    if '-model_des_save_path' in n_inp:
        ind1      = n_inp.index('-model_des_save_path')
        model_des_save_path = str(n_inp[ind1+1])
    else:
        model_des_save_path = './tmp/'

    print '[Setting Parameters]Trained Model Description saving path is ', model_des_save_path

    if '-learning_rate' in n_inp:
        ind1      = n_inp.index('-learning_rate')
        learning_rate = float(n_inp[ind1+1])
    else:
        learning_rate = 1e-3

    print '[Setting Parameters]Initial learning_rate is ', learning_rate

    if '-batch_size' in n_inp:
        ind1      = n_inp.index('-batch_size')
        batch_size = int(n_inp[ind1+1])
    else:
        batch_size = 10

    print '[Setting Parameters]Training batch_size is ', batch_size

    if '-num_epoch' in n_inp:
        ind1      = n_inp.index('-num_epoch')
        num_epoch = int(n_inp[ind1+1])
    else:
        num_epoch = 10

    print '[Setting Parameters]Training num_epoch is ', num_epoch

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
    print '[Setting Parameters] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback
    codec  = [trellis1, trellis2, interleaver]

    ##########################################
    # Setting Up RNN Model
    ##########################################

    start_time = time.time()

    model = load_model(learning_rate=learning_rate,rnn_type=rnn_type, block_len=block_len,
                       network_saved_path = starting_model_path, num_hidden_unit=num_hidden_unit,
                       interleave_array = p_array, dec_iter_num = dec_iter_num, loss=train_loss)

    end_time = time.time()
    print '[RNN decoder]loading RNN model takes ', str(end_time-start_time), ' secs'

    ##########################################
    # Setting Up Channel & Train SNR
    ##########################################
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr  = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    SNR = -10*np.log10(sigma_snr**2)

    noiser = [noise_type, sigma_snr, vv, radar_power, radar_prob, 10000.0 , snr_mix]
    start_time = time.time()

    X_feed_test, X_message_test = build_rnn_data_feed(num_block_test,  block_len, noiser, codec)
    X_feed_train,X_message_train= build_rnn_data_feed(num_block_train, block_len, noiser, codec)

    print X_feed_train.shape, X_message_train.shape

    def turbo_generator(features, labels, batch_size):

        batch_features = np.zeros((batch_size, block_len, 5))
        batch_labels = np.zeros((batch_size,block_len, 1))
        while True:
            for i in range(batch_size):
                # choose random index in features
                index= np.random.choice(features.shape[0], 1)
                batch_features[i] = features[index, :, :]
                batch_labels[i]   = labels[index, :, :]

            yield batch_features, batch_labels

    identity = str(np.random.random())
    save_path = model_save_path + identity+ '.h5'
    model.save_weights(save_path)

    print '[Warning] Save every epoch', identity


    save_cb = keras.callbacks.ModelCheckpoint('./tmp/save'+identity+ '_{epoch:02d}-{val_loss:.2f}' +'.h5', monitor='val_loss', verbose=0,
                                              save_best_only=False, save_weights_only=True, mode='auto', period=1)


    model.fit_generator(turbo_generator(X_feed_train, X_message_train, batch_size),
                        steps_per_epoch=num_block_train/batch_size,
                        epochs=num_epoch,
                        validation_data=turbo_generator(X_feed_test, X_message_test, batch_size),
                        validation_steps=num_block_test/batch_size,
                        callbacks=[save_cb],
                        use_multiprocessing = True)

    # model.fit(x=X_feed_train, y=X_message_train, batch_size=batch_size,
    #           #callbacks=[change_lr],
    #           epochs=num_epoch, validation_data=(X_feed_test, X_message_test))  # starts training


    print '[Training] saved model in ', save_path

    print '[Training]This is SNR', SNR ,'Training'
    pd       = model.predict(X_feed_test,batch_size = 100)
    err_rate = sum(sum(sum(abs(np.round(pd)-X_message_test))))*1.0/(X_message_test.shape[0]*X_message_test.shape[1])

    print model.evaluate(X_feed_test, X_message_test, batch_size=10)
    print '[Training]RNN turbo decoding has error rate ', err_rate
    end_time = time.time()
    print '[Trainiing]Training time is', str(end_time-start_time)


