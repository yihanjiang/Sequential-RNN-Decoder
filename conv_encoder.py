from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys

import keras
from keras.layers import Input, Embedding, LSTM,GRU, Dense, TimeDistributed, Lambda
from keras.models import Model
from keras.layers.wrappers import  Bidirectional

from keras.legacy import interfaces
from keras.optimizers import Optimizer


import keras.backend as K
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

from utils import code_err, conv_enc

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
frac = 0.45

config.gpu_options.per_process_gpu_memory_fraction = frac
set_session(tf.Session(config=config))
print '[Test][Warining] Restrict GPU memory usage to', frac, ', enable',str(int(1.0/frac)), 'processes'
import matplotlib.pyplot as plt

import numpy as np

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=5000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)

    parser.add_argument('-num_Enc_layer',  type=int, default=2)
    parser.add_argument('-num_Enc_unit',  type=int, default=100)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')
    parser.add_argument('-enc_direction', choices = ['bd', 'sd'], default = 'bd')

    parser.add_argument('-enc_activation', choices = ['sigmoid', 'linear', 'tanh'], default = 'tanh')
    parser.add_argument('-loss', choices = ['mse', 'binary_crossentropy'], default = 'binary_crossentropy')

    parser.add_argument('-batch_size',  type=int, default=100)
    parser.add_argument('-learning_rate',  type=float, default=0.01)
    parser.add_argument('-num_epoch',  type=int, default=5)


    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-enc_weight', type=str, default='default')

    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-enc3',  type=int, default=1)

    parser.add_argument('-feedback',  type=int, default=0)

    parser.add_argument('--GPU_proportion', type=float, default=1.00)

    args = parser.parse_args()
    print args

    print '[ID]', args.id
    return args

def build_encoder(args):

    ont_pretrain_trainable = True
    dropout_rate           = 1.0

    input_x         = Input(shape = (args.block_len, 1), dtype='float32', name='G_input')
    combined_x      = input_x
    for layer in range(args.num_Enc_layer):
        if args.enc_direction == 'bd':
            if args.rnn_setup == 'gru':
                combined_x = Bidirectional(GRU(units=args.num_Enc_unit, activation='tanh', dropout=dropout_rate,
                                               return_sequences=True, trainable=ont_pretrain_trainable),
                                           name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)
            else:
                combined_x = Bidirectional(LSTM(units=args.num_Enc_unit, activation='tanh', dropout=dropout_rate,
                                                return_sequences=True, trainable=ont_pretrain_trainable),
                                           name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)
        else:
            if args.rnn_setup == 'gru':
                combined_x = GRU(units=args.num_Enc_unit, activation='tanh', dropout=dropout_rate,
                                 return_sequences=True, trainable=ont_pretrain_trainable,
                                 name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)
            else:
                combined_x = LSTM(units=args.num_Enc_unit, activation='tanh', dropout=dropout_rate,
                                  return_sequences=True, trainable=ont_pretrain_trainable,
                                  name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)

        combined_x = BatchNormalization(name = 'G_bn_'+str(layer), trainable=ont_pretrain_trainable)(combined_x)

    encode = TimeDistributed(Dense(args.code_rate, activation=args.enc_activation),
                             trainable=ont_pretrain_trainable, name = 'G_fc')(combined_x)  #sigmoid

    return Model(input_x, encode)

def train(args):

    X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
    X_test_raw  = np.random.randint(0,2,args.block_len * args.num_block/args.test_ratio)

    X_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
    X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))

    X_conv_train = 2.0*conv_enc(X_train, args)-1.0
    X_conv_test  = 2.0*conv_enc(X_test, args) - 1.0

    model = build_encoder(args)

    if args.enc_weight == 'default':
        print 'Encoder has no weight'
    else:
        print 'Encoder loaded weight', args.Enc_weight
        model.load_weights(args.Enc_weight)


    optimizer = Adam(args.learning_rate)

    model.compile(loss=args.loss,  optimizer=optimizer, metrics=[code_err])
    model.summary()

    model.fit(X_train, X_conv_train, validation_data=(X_test, X_conv_test),
              batch_size=args.batch_size, epochs=args.num_epoch)

    model.save_weights('./tmp/conv_enc_'+args.id+'.h5')


if __name__ == '__main__':

    args = get_args()

    if args.GPU_proportion < 1.00:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        frac = args.GPU_proportion

        config.gpu_options.per_process_gpu_memory_fraction = frac
        set_session(tf.Session(config=config))
        print '[Test][Warining] Restrict GPU memory usage to 45%, enable',str(int(1/args.GPU_proportion)), 'processes'

    train(args)
