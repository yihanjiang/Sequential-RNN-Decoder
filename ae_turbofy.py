__author__ = 'yihanjiang'
'''
Under Turbo Structure with both encoder and decoder trainable
Under experiment, probably won't be public
'''

def get_args():
    pass


from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys

from keras.layers.convolutional import Conv1D

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

from utils import snr_db2sigma

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_block', type=int, default=10000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=1)

    parser.add_argument('-num_G_layer',  type=int, default=2)
    parser.add_argument('-num_G_unit',  type=int, default=100)

    parser.add_argument('-num_Dec_layer',  type=int, default=2)
    parser.add_argument('-num_Dec_unit',  type=int, default=200)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')

    parser.add_argument('-batch_size',  type=int, default=10)
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=8.0)
    parser.add_argument('-sharpen_code', type=int, default=1)
    parser.add_argument('-dropout_rate', type=float, default=1.0)

    parser.add_argument('-enc_act', choices=['sigmoid', 'linear'], default='linear')
    parser.add_argument('-loss', choices=['binary_crossentropy', 'mean_squared_error'], default='mean_squared_error')

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-G_weight', type=str, default='default')
    parser.add_argument('-Dec_weight', type=str, default='default')

    # parser.add_argument('-G_weight', type=str, default='./tmp/conv_enc_385460.h5')
    # parser.add_argument('-Dec_weight', type=str, default='./tmp/conv_dec439127.h5')

    args = parser.parse_args()
    print args

    print '[ID]', args.id
    return args

class AE():
    def __init__(self, args):

        optimizer = Adam(args.learning_rate)

        self.encoder = self.build_encoder(args)
        self.encoder.compile(loss=[args.loss], optimizer=optimizer, metrics=[errors])
        self.encoder.summary()

        if args.G_weight == 'default':
            print 'Encoder has no weight'
        else:
            print 'Encoder loaded weight', args.G_weight
            self.encoder.load_weights(args.G_weight)

        self.normalize = self.build_normalize(args)
        self.normalize.compile(loss=[args.loss], optimizer=optimizer, metrics=[errors])

        self.channel = self.build_channel(args)
        self.channel.compile(loss=[args.loss], optimizer=optimizer, metrics=[errors])


        self.decoder = self.build_decoder(args)
        self.decoder.compile(loss=[args.loss], optimizer=optimizer, metrics=[errors])
        self.decoder.summary()

        if args.Dec_weight == 'default':
            print 'Decoder has no weight'
        else:
            print 'Decoder loaded weight', args.Dec_weight
            self.decoder.load_weights(args.Dec_weight)

        x = Input(shape=(args.block_len, 1))

        code          = self.encoder(x)
        received_code = self.channel(self.normalize(code))
        x_hat = self.decoder(received_code)

        self.ae = Model(x, x_hat)
        self.ae.compile(loss=[args.loss], metrics=[errors], optimizer=optimizer)


    def build_interleaver(self,args):
        return None

    def build_deinterleaver(self, args):
        return None

    def build_normalize(self, args):
        def normalize(x):
            x_mean, x_var = tf.nn.moments(x,[0])
            x = (x-x_mean)*1.0/tf.sqrt(x_var)
            return x

        input_x      = Input(shape= (args.block_len, args.code_rate))
        input_x_norm = Lambda(normalize)(input_x)

        return Model(input_x, input_x_norm)

    def build_channel(self, args):

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

        input_x      = Input(shape= (args.block_len, args.code_rate))
        output_y     = Lambda(channel)(input_x)

        return Model(input_x, output_y)

    def build_encoder(self, args):

        ont_pretrain_trainable = True
        dropout_rate           = 1.0

        data_input = Input(shape = (args.block_len, 1))
        combined_x = data_input

        for layer in range(args.num_G_layer):
            if args.rnn_setup == 'gru':
                combined_x = Bidirectional(GRU(units=args.num_G_unit, activation='tanh', dropout=dropout_rate,
                                               return_sequences=True, trainable=ont_pretrain_trainable),
                                           name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)
            else:
                combined_x = Bidirectional(LSTM(units=args.num_G_unit, activation='tanh', dropout=dropout_rate,
                                                return_sequences=True, trainable=ont_pretrain_trainable),
                                           name = 'G_'+args.rnn_setup+'_'+str(layer))(combined_x)

            combined_x = BatchNormalization(name = 'G_bn_'+str(layer), trainable=ont_pretrain_trainable)(combined_x)

        fake_uw = TimeDistributed(Dense(args.code_rate, activation=args.enc_act, trainable=ont_pretrain_trainable),
                                  name = 'G_fc')(combined_x)

        return Model(data_input, fake_uw)

    def build_decoder(self, args):
        ont_pretrain_trainable = True
        dropout_rate           = 1.0

        received_input         = Input(shape = (args.block_len, args.code_rate), dtype='float32', name='Dec_input')
        combined_x             = received_input
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

        return Model(received_input, decode)

    def train(self, args):

        X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
        X_test_raw  = np.random.randint(0,2,args.block_len * args.num_block/args.test_ratio)

        X_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
        X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))


        self.test(args, args.G_weight,args.Dec_weight )

        if args.G_weight!='default':
            self.ae.load_weights(args.G_weight, by_name=True)
        if args.Dec_weight!='default':
            self.ae.load_weights(args.Dec_weight, by_name=True)

        self.ae.fit(X_train, X_train, validation_data=(X_test, X_test),
                    batch_size=args.batch_size, epochs=args.num_epoch)

        self.ae.save_weights('./tmp/enc1'+args.id+'.h5')

        self.test(args, './tmp/enc1'+args.id+'.h5')


    def test(self, args, g_weight = 'default', d_weight = 'default'):

        ###################################################
        # Testing
        ###################################################

        X_test_raw  = np.random.randint(0,2,args.num_block*args.block_len/args.test_ratio)
        X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))

        #print 'Testing before fine-tuning'
        snr_start = -1.0
        snr_stop  = 8.0
        snr_points = 10

        dropout_rate = 1.0
        enc_trainable = False
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

            inputs = Input(shape=(args.block_len, 1))
            x = inputs

            def channel(x):
                noise_sigma =  snr_db2sigma(snr_db)
                return x+ noise_sigma*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

            def normalize(x):
                x_mean, x_var = tf.nn.moments(x,[0])
                x = (x-x_mean)*1.0/tf.sqrt(x_var)

                if args.sharpen_code:
                    x = (100000.0 * x) + 0.0
                    x = K.clip(x, 0.0, 1.0)
                    x = 2*x - 1
                return x

            for layer in range(args.num_G_layer):

                if args.rnn_setup == 'lstm':
                    x = Bidirectional(LSTM(units=args.num_G_unit, activation='tanh', return_sequences=True,
                                         dropout=dropout_rate), trainable=enc_trainable, name = 'G_'+args.rnn_setup+'_'+str(layer))(x)
                elif args.rnn_setup == 'gru':
                    x = Bidirectional(GRU(units=args.num_G_unit, activation='tanh', return_sequences=True,
                                         dropout=dropout_rate), trainable=enc_trainable, name = 'G_'+args.rnn_setup+'_'+str(layer))(x)
                elif args.rnn_setup == 'fc':
                    x = Dense(args.num_G_unit,activation='elu' ,trainable=enc_trainable)(x)
                elif args.rnn_setup == 'cnn':
                    x = Conv1D(args.num_G_unit, args.conv_kernel_size, strides=1, padding='same',name = 'G_'+args.rnn_setup+'_'+str(layer),
                               activation='elu',trainable=enc_trainable)(x)

                x = BatchNormalization(trainable=enc_trainable, name = 'G_bn_'+str(layer))(x)

            enc_output = TimeDistributed(Dense(args.code_rate, activation=args.enc_act),
                                         name = 'G_fc',
                                         trainable=enc_trainable)(x)

            encoder    = Lambda(normalize)(enc_output)
            x          = Lambda(channel)(encoder)

            for layer in range(args.num_Dec_layer - 1):
                if args.rnn_setup == 'lstm':
                    x = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                         dropout=dropout_rate), trainable=dec_trainable, name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)
                elif args.rnn_setup == 'gru':
                    x = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                         dropout=dropout_rate), trainable=dec_trainable, name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)

                x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(layer))(x)

            y = x

            if args.rnn_setup == 'lstm':
                y = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                     dropout=args.dropout_rate), trainable=dec_trainable, name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)
            elif args.rnn_setup == 'gru':
                y = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                     dropout=args.dropout_rate), trainable=dec_trainable, name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)

            x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(args.num_Dec_layer-1))(y)

            predictions = TimeDistributed(Dense(1, activation='sigmoid'), trainable=dec_trainable, name = 'Dec_fc')(x)

            model_test = Model(inputs=inputs, outputs=predictions)

            model_test.compile(optimizer=keras.optimizers.adam(),loss=args.loss, metrics=[errors])

            if g_weight!='default':
                model_test.load_weights(g_weight, by_name=True)

            if d_weight!='default':
                model_test.load_weights(d_weight, by_name=True)

            pd       = model_test.predict(X_test, verbose=0)
            decoded_bits = np.round(pd)
            ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_test))))*1.0/(X_test.shape[0]*X_test.shape[1])# model.evaluate(X_feed_test, X_message_test, batch_size=10)
            tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
            bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

            ber.append(ber_err_rate)
            bler.append(bler_err_rate)

            del model_test

        print 'SNRS:', SNRS_dB
        print 'BER:',ber
        print 'BLER:',bler



if __name__ == '__main__':
    args = get_args()

    aae = AE(args)
    aae.train(args)
