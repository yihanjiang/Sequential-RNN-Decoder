__author__ = 'yihanjiang'

from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda

from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras import regularizers

import sys
import pickle
import numpy as np
import math

from bcjr_util import generate_bcjr_example

if __name__ == '__main__':
    # Arguments
    n_inp = sys.argv[1:]
    if '--help' in n_inp:
        with open('./manuals/train_bcjr_rnn.md','r') as fin:
            print fin.read()
        sys.exit()

    print '[BCJR] Training BCJR-like RNN'


    if '-noise_type' in n_inp:
        ind1      = n_inp.index('-noise_type')
        noise_type = str(n_inp[ind1+1])
    else:
        noise_type = 'awgn'

    vv          = -1.0
    radar_power = -1.0
    radar_prob  = -1.0

    if noise_type == 'awgn':
        print '[BCJR Setting Parameters] Noise Type is ', noise_type

    elif noise_type == 't-dist':
        if '-v' in n_inp:
            ind1 = n_inp.index('-v')
            vv   = float(n_inp[ind1+1])
        else:
            vv   = 5.0
        print '[BCJR Setting Parameters] Noise Type is ', noise_type, ' with v=', vv

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

        print '[BCJR Setting Parameters] Noise Type is ', noise_type, 'with Radar Power ', radar_power, ' with Radar Probability ', radar_prob


    if '-network_model_path' in n_inp:
        ind1      = n_inp.index('-network_model_path')
        starting_model_path = str(n_inp[ind1+1])
        is_pretrained = True
    else:
        starting_model_path = './tmp/bcjr_train100_truePostLL_0.261448005038_1.h5'
        is_pretrained = False

    print '[BCJR Setting Parameters][Not functional] Network starting path is ', starting_model_path

    if '-model_save_path' in n_inp:
        ind1      = n_inp.index('-model_save_path')
        model_save_path = str(n_inp[ind1+1])
    else:
        model_save_path = './tmp/'

    print '[BCJR Setting Parameters] Trained Model Weight saving path is ', model_save_path

    if '-learning_rate' in n_inp:
        ind1      = n_inp.index('-learning_rate')
        learning_rate = float(n_inp[ind1+1])
    else:
        learning_rate = 1e-2

    print '[BCJR Setting Parameters] Initial learning_rate is ', learning_rate

    if '-batch_size' in n_inp:
        ind1      = n_inp.index('-batch_size')
        batch_size = float(n_inp[ind1+1])
    else:
        batch_size = 10

    print '[BCJR Setting Parameters] Training batch_size is ', batch_size

    if '-num_epoch' in n_inp:
        ind1      = n_inp.index('-num_epoch')
        num_epoch = float(n_inp[ind1+1])
    else:
        num_epoch = 10

    print '[BCJR Setting Parameters] Training num_epoch is ', num_epoch

    if '-num_iteration' in n_inp:
        ind1      = n_inp.index('-num_iteration')
        num_iteration = int(n_inp[ind1+1])
    else:
        num_iteration = 6

    print '[BCJR Setting Parameters] Turbo Decoding Iteration ', num_iteration

    import commpy.channelcoding.interleavers as RandInterlv
    import commpy.channelcoding.convcode as cc

    if '-codec_type' in n_inp:
        ind1           = n_inp.index('-codec_type')
        codec_type = str(n_inp[ind1+1])
    else:
        codec_type = 'default'
    print '[BCJR Setting Parameters] Codec type is ', codec_type

    if '--generate_example' in n_inp:
        generate_example = True
    else:
        generate_example = False

    print '[BCJR Setting Parameters] Is Generating Example ', generate_example

    if generate_example == False:
        if '-train_example_path' in n_inp:
            ind1      = n_inp.index('-train_example_path')
            train_example_path = str(ind1)
        else:
            train_example_path = './tmp/train_example_0915.pickle'

        if '-test_example_path' in n_inp:
            ind1      = n_inp.index('-test_example_path')
            test_example_path = str(ind1)
        else:
            test_example_path = './tmp/test_example_0915.pickle'

        with open(train_example_path) as f:
            bcjr_outputs_train, bcjr_inputs_train, num_iteration_train, block_len = pickle.load(f)

        with open(test_example_path) as f:
            bcjr_outputs_test,  bcjr_inputs_test,  num_iteration_test,  block_len= pickle.load(f)

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

    else:
        if '-train_snr' in n_inp:
            ind1      = n_inp.index('-train_snr')
            train_snr = int(n_inp[ind1+1])
        else:
            train_snr = 0.0

        print '[BCJR Setting Parameters] Training Data SNR is ', train_snr, ' dB'

        if '-block_len' in n_inp:
            ind1      = n_inp.index('-block_len')
            block_len = int(n_inp[ind1+1])
        else:
            block_len = 100
        print '[BCJR Setting Parameters] Code Block Length is ', block_len

        if '-num_block_train' in n_inp:
            ind1            = n_inp.index('-num_block_train')
            num_block_train = int(n_inp[ind1+1])
        else:
            num_block_train = 2000

        if '-num_block_test' in n_inp:
            ind1           = n_inp.index('-num_block_test')
            num_block_test = int(n_inp[ind1+1])
        else:
            num_block_test = 100

        print '[BCJR Setting Parameters] Number of Train Block is ', num_block_train, ' Test Block ', num_block_test

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

        bcjr_inputs_train, bcjr_outputs_train = generate_bcjr_example(num_block_train, block_len,
                                                                      codec, is_save = False,num_iteration = 6,
                                                                      train_snr_db = train_snr, save_path = './tmp/')

        bcjr_inputs_test,  bcjr_outputs_test  = generate_bcjr_example(num_block_test, block_len,
                                                                      codec, is_save = False, num_iteration = 6,
                                                                      train_snr_db = train_snr, save_path = './tmp/')

    if '-rnn_direction' in n_inp:
        ind1            = n_inp.index('-rnn_direction')
        rnn_direction = str(n_inp[ind1+1])
    else:
        rnn_direction = 'bd'
    print '[BCJR Setting Parameters] RNN Direction is ', rnn_direction

    if '-rnn_type' in n_inp:
        ind1            = n_inp.index('-rnn_type')
        rnn_type = str(n_inp[ind1+1])
    else:
        rnn_type = 'rnn-lstm'
    print '[BCJR Setting Parameters] RNN Model Type is ', rnn_type

    if '-num_rnn_layer' in n_inp:
        ind1            = n_inp.index('-num_rnn_layer')
        num_rnn_layer = int(n_inp[ind1+1])
    else:
        num_rnn_layer = 2
    print '[BCJR Setting Parameters] Number of RNN layer is ', num_rnn_layer

    if '-num_hunit_rnn' in n_inp:
        ind1            = n_inp.index('-num_hunit_rnn')
        num_hunit_rnn = int(n_inp[ind1+1])
    else:
        num_hunit_rnn = 200
    print '[BCJR Setting Parameters] Number of RNN unit is ', num_hunit_rnn

    train_batch_size  = 100                   # 100 good.
    test_batch_size   = 100
    dropout_rate      = 1.0                   # Dropout !=1.0 doesn't work!
    input_feature_num = 3

    # print parameters
    print '*'*100
    print '[BCJR] The RNN has ',rnn_direction,  'with ',num_rnn_layer, 'layers with ', num_hunit_rnn, ' unit'
    print '[BCJR] learning rate is ', learning_rate
    print '[BCJR] batch size is ', train_batch_size
    print '[BCJR] block length is ', block_len
    print '[BCJR] dropout is ', dropout_rate
    print '*'*100

    regularizer = regularizers.l2(0.000)

    inputs = Input(shape=(block_len, input_feature_num))
    Rx_received = inputs

    # Rx Decoder
    if rnn_type == 'rnn-lstm' or rnn_type == 'rnn':
        x = Rx_received
        for layer in range(num_rnn_layer):
            if rnn_direction == 'bd':
                x = Bidirectional(LSTM(units=num_hunit_rnn, activation='tanh',
                                       kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                                       return_sequences=True, dropout=dropout_rate))(x)
                x = BatchNormalization()(x)
            else:
                x = LSTM(units=num_hunit_rnn, activation='tanh',
                         kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                         return_sequences=True, dropout=dropout_rate)(x)
                x = BatchNormalization()(x)

    elif rnn_type == 'rnn-gru':
        x = Rx_received
        for layer in range(num_rnn_layer):
            if rnn_direction == 'bd':
                x = Bidirectional(GRU(units=num_hunit_rnn, activation='tanh',
                                      kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                                      return_sequences=True, dropout=dropout_rate))(x)
                x = BatchNormalization()(x)
            else:
                x = GRU(units=num_hunit_rnn, activation='tanh',
                        kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                        return_sequences=True, dropout=dropout_rate)(x)
                x = BatchNormalization()(x)

    elif rnn_type == 'simple-rnn':
        x = Rx_received
        for layer in range(num_rnn_layer):
            if rnn_direction == 'bd':
                x = Bidirectional(SimpleRNN(units=num_hunit_rnn, activation='tanh',
                                      kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                                      return_sequences=True, dropout=dropout_rate))(x)
                x = BatchNormalization()(x)
            else:
                x = SimpleRNN(units=num_hunit_rnn, activation='tanh',
                        kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                        return_sequences=True, dropout=dropout_rate)(x)
                x = BatchNormalization()(x)


    else:
        print 'not supported'
        sys.exit()

    def errors(y_true, y_pred):
        myOtherTensor = K.not_equal(K.round(y_true), K.round(y_pred))
        return K.mean(tf.cast(myOtherTensor, tf.float32))

    predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_regularizer=regularizer))(x)

    model = Model(inputs=inputs, outputs=predictions)
    optimizer= keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])
    #print(model.summary())

    ######################
    # Build Data Format
    #####################
    # print bcjr_inputs_train.shape
    # print bcjr_outputs_train.shape

    bcjr_inputs_train   = bcjr_inputs_train.reshape((-1, block_len, input_feature_num))
    bcjr_outputs_train  = bcjr_outputs_train.reshape((-1,  block_len, 1))

    # print bcjr_inputs_train.shape
    # print bcjr_outputs_train.shape

    # output is not sum, need to debug here.
    if 1==1:
        target_train_select = bcjr_outputs_train[:,:,0] + bcjr_inputs_train[:,:,2]
    else:
        target_train_select = bcjr_outputs_train[:,:,0]


    target_train_select[:,:] = math.e**target_train_select[:,:]*1.0/(1+math.e**target_train_select[:,:])

    X_input  = bcjr_inputs_train.reshape(-1,block_len,3)
    X_target = target_train_select.reshape(-1,block_len,1)

    #X_target[:,:] = math.e**X_target[:,:]*1.0/(1+math.e**X_target[:,:])

    print X_input.shape
    print X_target.shape

    ###########################
    # Saving Logs to positions
    ###########################
    identity = str(np.random.random())
    with open('./tmp/log_'+identity+'.txt','w') as f:
        f.write('*'*100+'\n')
        f.write('* model for '+identity+'\n')
        f.write('*'*100+'\n')
        f.write('Hello World')
        f.close()

    print '[BCJR] Model Log saved in ', './tmp/log_'+identity+'.txt'

    ###########################
    # Start Training
    ###########################


    if is_pretrained:
        model.load_weights(starting_model_path)
        print '[BCJR][Warning] Loaded Some init weight', starting_model_path
    else:
        print '[BCJR][Warning] Train from scratch, not loading weight!'

    model.fit(x=X_input, y=X_target, batch_size=train_batch_size,
              epochs=num_epoch, validation_split = 0.1)#validation_data=(test_tx, X_test))  # starts training

    model_des = str(codec_type) +'_'+ str(num_hunit_rnn)+'_' + str(rnn_type)+'_' + str(rnn_direction)+'_' +str(block_len)
    model.save_weights('./tmp/bcjr_train'+str(model_des)+str(identity)+'_1.h5')
    print '[BCJR] Saved Model at', './tmp/bcjr_train'+str(model_des)+str(identity)+'_1.h5'

    del model

    model = Model(inputs=inputs, outputs=predictions)
    optimizer= keras.optimizers.adam(lr=learning_rate/10.0)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])

    model.fit(x=X_input, y=X_target, batch_size=train_batch_size,
              epochs=num_epoch, validation_split = 0.1)#validation_data=(test_tx, X_test))  # starts training

    model.save_weights('./tmp/bcjr_train'+str(model_des)+str(identity)+'_2.h5')
    print '[BCJR] Saved Model at', './tmp/bcjr_train'+str(model_des)+str(identity)+'_2.h5'

