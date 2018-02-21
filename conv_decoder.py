from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
import keras
from keras.layers import Input, Embedding, LSTM,GRU, Dense, TimeDistributed, Lambda
from keras.models import Model
from keras.layers.wrappers import  Bidirectional

from keras.legacy import interfaces
from keras.optimizers import Optimizer
import commpy.channelcoding.convcode as cc

import keras.backend as K
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
frac = 0.45

config.gpu_options.per_process_gpu_memory_fraction = frac
set_session(tf.Session(config=config))
print '[Test][Warining] Restrict GPU memory usage to', frac, ', enable',str(int(1.0/frac)), 'processes'
import matplotlib.pyplot as plt

import numpy as np

def conv_enc(X_train_raw, args):
    num_block = X_train_raw.shape[0]
    block_len = X_train_raw.shape[1]
    x_code    = []

    generator_matrix = np.array([[args.enc1, args.enc2]])
    M = np.array([args.M]) # Number of delay elements in the convolutional encoder
    trellis = cc.Trellis(M, generator_matrix,feedback=args.feedback)# Create trellis data structure

    for idx in range(num_block):
        xx = cc.conv_encode(X_train_raw[idx, :, 0], trellis)
        xx = xx[2*int(M):]
        xx = xx.reshape((block_len, 2))

        x_code.append(xx)

    return np.array(x_code)

def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def snr_db2sigma(train_snr):
    block_len    = 100
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=5000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)

    parser.add_argument('-num_Dec_layer',  type=int, default=2)
    parser.add_argument('-num_Dec_unit',  type=int, default=500)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')

    parser.add_argument('-batch_size',  type=int, default=10)
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")

    parser.add_argument('-loss', choices = ['binary_crossentropy', 'mean_squared_error'], default = 'mean_squared_error')

    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=8.0)

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-Dec_weight', type=str, default='default')


    args = parser.parse_args()
    print args

    print '[ID]', args.id
    return args

def build_decoder(args):

    ont_pretrain_trainable = True
    dropout_rate           = 1.0

    def channel(x):
        print 'training with noise snr db', args.train_channel_low, args.train_channel_high
        noise_sigma_low =  snr_db2sigma(args.train_channel_low) # 0dB
        noise_sigma_high =  snr_db2sigma(args.train_channel_high) # 0dB
        print 'training with noise snr db', noise_sigma_low, noise_sigma_high
        noise_sigma =  tf.random_uniform(tf.shape(x),
            minval=noise_sigma_high,
            maxval=noise_sigma_low,
            dtype=tf.float32
        )

        return x+ noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

    input_x         = Input(shape = (args.block_len, args.code_rate), dtype='float32', name='D_input')
    combined_x      = Lambda(channel)(input_x)

    for layer in range(args.num_Dec_layer):
        if args.rnn_setup == 'gru':
            combined_x = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                           return_sequences=True, trainable=ont_pretrain_trainable),
                                       name = 'Dec_'+args.rnn_setup+'_'+str(layer))(combined_x)
        else:
            combined_x = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                            return_sequences=True, trainable=ont_pretrain_trainable),
                                       name = 'Dec_'+args.rnn_setup+'_'+str(layer))(combined_x)

        combined_x = BatchNormalization(name = 'Dec_bn'+'_'+str(layer), trainable=ont_pretrain_trainable)(combined_x)

    decode = TimeDistributed(Dense(1, activation='sigmoid'), trainable=ont_pretrain_trainable, name = 'Dec_fc')(combined_x)  #sigmoid

    return Model(input_x, decode)

def train(args):

    X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
    X_test_raw  = np.random.randint(0,2,args.block_len * args.num_block/args.test_ratio)

    X_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
    X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))

    X_conv_train = 2.0*conv_enc(X_train, args) - 1.0
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    model = build_decoder(args)

    def scheduler(epoch):

        if epoch > 10 and epoch <=15:
            print 'changing by /10 lr'
            lr = args.learning_rate/10.0
        elif epoch >15 and epoch <=20:
            print 'changing by /100 lr'
            lr = args.learning_rate/100.0
        elif epoch >20 and epoch <=25:
            print 'changing by /1000 lr'
            lr = args.learning_rate/1000.0
        elif epoch > 25:
            print 'changing by /10000 lr'
            lr = args.learning_rate/10000.0
        else:
            lr = args.learning_rate

        return lr
    change_lr = LearningRateScheduler(scheduler)


    if args.Dec_weight == 'default':
        print 'Decoder has no weight'
    else:
        print 'Decoder loaded weight', args.Dec_weight
        model.load_weights(args.Dec_weight)


    optimizer = Adam(args.learning_rate)

    # Build and compile the discriminator
    model.compile(loss=args.loss,  optimizer=optimizer, metrics=[errors])
    model.summary()

    model.fit(X_conv_train,X_train, validation_data=(X_conv_test, X_test),
              callbacks = [change_lr],
              batch_size=args.batch_size, epochs=args.num_epoch)

    model.save_weights('./tmp/conv_dec'+args.id+'.h5')

def test(args, dec_weight):
    X_test_raw  = np.random.randint(0,2,args.num_block*args.block_len/args.test_ratio)
    X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    #print 'Testing before fine-tuning'
    snr_start = -1.0
    snr_stop  = 8
    snr_points = 10

    dec_trainable = True

    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(float(args.num_block)/float(args.num_block*2.0)) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print '[testing]', SNRS_dB

    ber, bler = [],[]
    for idx, snr_db in enumerate(SNRS_dB):

        inputs = Input(shape=(args.block_len, args.code_rate))

        def channel(x):
            noise_sigma =  snr_db2sigma(snr_db)
            return x+ noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

        x          = Lambda(channel)(inputs)

        for layer in range(args.num_Dec_layer - 1):
            if args.rnn_setup == 'lstm':
                x = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                     trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)
            elif args.rnn_setup == 'gru':
                x = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                     trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)

            x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(layer))(x)

        y = x

        if args.rnn_setup == 'lstm':
            y = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)
        elif args.rnn_setup == 'gru':
            y = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)

        x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(args.num_Dec_layer-1))(y)

        predictions = TimeDistributed(Dense(1, activation='sigmoid'), trainable=dec_trainable, name = 'Dec_fc')(x)

        model_test = Model(inputs=inputs, outputs=predictions)

        model_test.compile(optimizer=keras.optimizers.adam(),loss=args.loss, metrics=[errors])

        model_test.load_weights(dec_weight, by_name=True)

        pd       = model_test.predict(X_conv_test, verbose=0)
        decoded_bits = np.round(pd)
        ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_test))))*1.0/(X_test.shape[0]*X_test.shape[1])# model.evaluate(X_feed_test, X_message_test, batch_size=10)
        tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
        #
        # print ber_err_rate
        # print bler_err_rate

        ber.append(ber_err_rate)
        bler.append(bler_err_rate)

        del model_test

    print 'SNRS:', SNRS_dB
    print 'BER:',ber
    print 'BLER:',bler





if __name__ == '__main__':

    args = get_args()

    train(args)
    test(args, dec_weight='./tmp/conv_dec'+args.id+'.h5')