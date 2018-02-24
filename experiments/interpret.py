__author__ = 'yihanjiang'

'''
Interpretibility Module.Not a command line tool!
'''

import math
import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv
import commpy.channelcoding.turbo as turbo
import matplotlib.pylab as plt

import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
import keras
from keras.layers.wrappers import  Bidirectional


from utils import corrupt_signal, build_rnn_data_feed

class Interpret(object):
    def __init__(self,network_saved_path,
                 num_block = 100,rnn_type = 'lstm',
                 block_len = 100,num_hidden_unit = 200, is_ll = True, no_bn = False):

        self.block_len = block_len
        self.num_block = num_block
        self.network_saved_path = network_saved_path
        self.rnn_type = rnn_type
        if is_ll:
            self.model  = self._load_model(no_bn = no_bn)


    def _load_model(self,  num_hidden_unit = 200, no_bn =False):

        '''
        Only works for interpretibility, you have to know: num_hidden_unit.
        :param block_len:
        :param network_saved_path:
        :param learning_rate:
        :param num_hidden_unit:
        :return:
        '''
        if self.network_saved_path == 'default':
            network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
        else:
            network_saved_path = self.network_saved_path

        rnn_type    = self.rnn_type   #'gru', 'lstm'
        print '[BCJR RNN Interpret] using model type', rnn_type
        print '[BCJR RNN Interpret] using model path', network_saved_path

        batch_size    = 32

        print '[RNN Model] Block length', self.block_len
        print '[RNN Model] Evaluate Batch size', batch_size

        ####################################################
        # Define Model
        ####################################################
        if rnn_type == 'lstm':
            f1 = Bidirectional(LSTM(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(LSTM(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')
        else:
            f1 = Bidirectional(GRU(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(GRU(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')

        f5 = TimeDistributed(Dense(1, activation='linear'),name='time_distributed_1')

        inputs = Input(shape = (self.block_len,3))
        if no_bn:
            predictions = f5(f3(f1(inputs)))
        else:
            predictions = f5(f4(f3(f2(f1(inputs)))))

        model = Model(inputs=inputs, outputs=predictions)
        optimizer= keras.optimizers.adam(lr=0.001, clipnorm=1.0)               # not useful
        model.compile(optimizer=optimizer,loss='mean_squared_error')
        #print model.summary()
        #

        model.load_weights(network_saved_path, by_name=True)
        # if rnn_type == 'lstm':
        #     model.load_weights(network_saved_path, by_name=True)
        # else:
        #     model.load_weights(network_saved_path)
        #     model.save_weights('./tmp/gruuu.h5')



        return model

    # Helper function for adding bursty noise
    def add_bursty_noise(self, X_feed_test, bit_pos, radar_noise_power, is_interleave = False, interleave_array = None):

        if is_interleave == False:
            radar_noise = np.zeros(X_feed_test.shape)
            radar_noise[:, bit_pos, :] = radar_noise_power

        else:
            radar_noise = np.zeros(X_feed_test.shape)
            inter_pos   = interleave_array[bit_pos]
            radar_noise[:, bit_pos, 0] = radar_noise_power
            radar_noise[:, bit_pos, 1] = radar_noise_power
            radar_noise[:, inter_pos, 3] = radar_noise_power
            radar_noise[:, inter_pos, 4] = radar_noise_power

        X_feed_test += radar_noise

        return X_feed_test

    def likelihood(self, bit_pos_list, sigma, radar_noise_power=20.0,
                   is_same_code = False, is_all_zero = True,
                   is_compute_no_bursty = False, is_compute_map = False,
                   normalize_input = False, is_t = False):
        '''
        compute the likelihood along the positions.
        :param bit_pos:
        :param sigma:
        :param radar_noise_power:
        :param is_compute_no_bursty:
        :param is_compute_map:
        :return:
        '''

        model = self.model

        # Define the code, too zy, reuse the code
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])
        feedback = 7

        trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        interleaver = RandInterlv.RandInterlv(self.block_len, 0)

        if is_t == False:
            noiser = ['awgn', sigma]
        else:
            noiser = ['t-dist', sigma, 3, -1, -1, -1, [-1, -1, -1]]
            print noiser

        codec  = [trellis1, trellis2, interleaver]
        X_feed_test, X_message_test = build_rnn_data_feed(self.num_block, self.block_len, noiser, codec,
                                                          is_same_code=is_same_code, is_all_zero=is_all_zero)
        if normalize_input:
            X_feed_test /= (sigma**2)

        if is_compute_no_bursty:
            ##############################################################
            # Non Bursty Noise Case
            # RNN Decoder
            ##############################################################
            pd                = model.predict(X_feed_test[:, :, :3],batch_size = 100)
            rnn_ll_non_bursty = np.mean(pd, axis = 0).T.tolist()[0]

            if is_compute_map:
                # BCJR/MAP Decoder
                map_likelihood_list = []
                for block_idx in range(self.num_block):
                    message_bits =  X_message_test[block_idx, :, :]
                    encoded      =  X_feed_test[block_idx, :, :2]
                    sys_r        =  encoded[:, 0]
                    par1_r       =  encoded[:, 1]

                    L_int = np.zeros(len(sys_r))

                    if normalize_input:
                        [L_ext_1_noburst, decoded_bits] = turbo.map_decode(sys_r, par1_r,
                                                             trellis1, 1.0, L_int, 'decode')

                    else:
                        [L_ext_1_noburst, decoded_bits] = turbo.map_decode(sys_r, par1_r,
                                                             trellis1, sigma**2, L_int, 'decode')

                    map_likelihood_list.append(L_ext_1_noburst)

                map_ll_non_bursty = np.stack(np.array(map_likelihood_list), axis=0)
                map_ll_non_bursty = np.mean(map_ll_non_bursty, axis=0).T.tolist()

        for bit_pos in bit_pos_list:
            X_feed_test = self.add_bursty_noise(X_feed_test, bit_pos, radar_noise_power)

        # RNN Decoder
        pd     = model.predict(X_feed_test[:, :, :3],batch_size = 100)
        rnn_ll_bursty = np.mean(pd, axis = 0).T.tolist()[0]

        if is_compute_map:
            # BCJR/MAP Decoder
            map_likelihood_list = []
            for block_idx in range(self.num_block):
                message_bits =  X_message_test[block_idx, :, :]
                encoded      =  X_feed_test[block_idx, :, :2]
                sys_r        =  encoded[:, 0]
                par1_r       =  encoded[:, 1]

                # CommPy Map
                L_int = np.zeros(len(sys_r))

                [L_ext_1_noburst, decoded_bits] = turbo.map_decode(sys_r, par1_r,
                                                     trellis1, sigma**2,  L_int, 'decode')

                map_likelihood_list.append(L_ext_1_noburst)

            map_ll_bursty = np.stack(np.array(map_likelihood_list), axis=0)
            map_ll_bursty = np.mean(map_ll_bursty, axis=0).T.tolist()

        if is_compute_no_bursty:
            if is_compute_map:
                return map_ll_non_bursty, rnn_ll_non_bursty, map_ll_bursty, rnn_ll_bursty
            else:
                return rnn_ll_non_bursty, rnn_ll_bursty
        else:
            if is_compute_map:
                return map_ll_bursty, rnn_ll_bursty
            else:
                return rnn_ll_bursty

    def ber(self, bit_pos_list, sigma, radar_noise_power=20.0, is_compute_no_bursty = False, is_compute_map = False,
             is_t = False, is_compute_bursty = True):
        '''
        Compute Turbo BER along block in different positions.
        '''

        K.clear_session()
        # Define the code, too zy, reuse the code
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])
        feedback = 7

        trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        interleaver = RandInterlv.RandInterlv(self.block_len, 0)

        from turbo_rnn import load_model

        dec_iter_num    = 6
        num_hidden_unit = 200

        print '[BCJR RNN Interpret] number of block is', self.num_block

        model = load_model(interleave_array=interleaver.p_array, network_saved_path = self.network_saved_path, block_len=self.block_len,
                           dec_iter_num = dec_iter_num, num_hidden_unit=num_hidden_unit)

        if is_t:
            noiser = ['awgn', sigma]
        else:
            noiser = ['t-dist', sigma, 3, -1, -1, -1, [-1, -1, -1]]
            print noiser

        codec  = [trellis1, trellis2, interleaver]
        X_feed_test, X_message_test = build_rnn_data_feed(self.num_block, self.block_len, noiser, codec, is_all_zero=False)

        if is_compute_no_bursty:
            # RNN Decoder
            pd     = model.predict(X_feed_test,batch_size = 100)
            rnn_ber_non_bursty = np.mean(np.round(pd)!=X_message_test, axis = 0).T.tolist()[0]

            # Turbo Decoder
            map_ber_list = []
            for block_idx in range(self.num_block):
                message_bits =  X_message_test[block_idx, :, :]
                encoded      =  X_feed_test[block_idx, :, :]
                sys_r        =  encoded[:, 0]
                par1_r       =  encoded[:, 1]
                par2_r       =  encoded[:, 4]

                decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, sigma**2, 6, interleaver, L_int = None)

                map_ber_list.append(np.array(decoded_bits) != message_bits.T)

            map_ber_non_bursty = np.stack(map_ber_list, axis=0)
            map_ber_non_bursty = np.mean(map_ber_non_bursty, axis=0).tolist()[0]

            print '[BCJR RNN Interpret] RNN BER for Non Bursty Noise Case', rnn_ber_non_bursty
            print '[BCJR RNN Interpret] BCJR BER for Non Bursty Noise Case', map_ber_non_bursty

        if is_compute_bursty:
            for bit_pos in bit_pos_list:
                X_feed_test = self.add_bursty_noise(X_feed_test, bit_pos, radar_noise_power,
                                                    is_interleave=True, interleave_array=interleaver.p_array)

            pd  = model.predict(X_feed_test,batch_size = 100)
            rnn_ber_bursty = np.mean(np.round(pd)!=X_message_test, axis = 0).T.tolist()[0]

            # Turbo Decoder
            map_ber_list = []
            for block_idx in range(self.num_block):
                message_bits =  X_message_test[block_idx, :, :]
                encoded      =  X_feed_test[block_idx, :, :]
                sys_r        =  encoded[:, 0]
                par1_r       =  encoded[:, 1]
                par2_r       =  encoded[:, 4]

                decoded_bits = turbo.hazzys_turbo_decode(sys_r, par1_r, par2_r, trellis1, sigma**2, 6, interleaver, L_int = None)

                map_ber_list.append(np.array(decoded_bits) != message_bits.T)

            map_ber_bursty = np.stack(map_ber_list, axis=0)
            map_ber_bursty = np.mean(map_ber_bursty, axis=0).tolist()[0]

            print '[BCJR RNN Interpret] RNN BER for Bursty Noise Case', rnn_ber_bursty
            print '[BCJR RNN Interpret] BCJR BER for Bursty Noise Case', map_ber_bursty
        #


        if is_compute_no_bursty and is_compute_bursty:
            if is_compute_map:
                return map_ber_non_bursty, rnn_ber_non_bursty, map_ber_bursty, rnn_ber_bursty
            else:
                return rnn_ber_non_bursty, rnn_ber_bursty
        elif is_compute_no_bursty == False and is_compute_bursty == True:
            if is_compute_map:
                return map_ber_bursty, rnn_ber_bursty
            else:
                return rnn_ber_bursty
        elif is_compute_no_bursty == True and is_compute_bursty == False:
            if is_compute_map:
                return map_ber_non_bursty, rnn_ber_non_bursty
            else:
                return rnn_ber_non_bursty





def likelihood_snr_range():
    print '[Interpret] Likelihood output of BCJR/RNN'
    ###############################################
    # Input Parameters
    ###############################################
    # network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
    # interpret = Interpret(network_saved_path=network_saved_path, block_len=100, num_block=100)
    normalize_input = True # 1012 test case
    network_saved_path = './model_zoo/nobn_awgn/test1.h5'
    interpret = Interpret(network_saved_path=network_saved_path, block_len=100, num_block=100, no_bn=False, rnn_type='gru')

    #network_saved_path = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_1.h5'
    #interpret = Interpret(network_saved_path=network_saved_path, block_len=100, num_block=100)

    radar_bit_pos = 50

    map_ll_non_bursty1, rnn_ll_non_bursty1, map_ll_bursty1, rnn_ll_bursty1 = interpret.likelihood(bit_pos_list=[radar_bit_pos], sigma=0.5,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True,
                                                                                              is_same_code = False, is_all_zero = True,
                                                                                              normalize_input = normalize_input)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True,
                                                                                              is_same_code = False, is_all_zero = True,
                                                                                              normalize_input = normalize_input)

    map_ll_non_bursty3, rnn_ll_non_bursty3, map_ll_bursty3, rnn_ll_bursty3 = interpret.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True,
                                                                                              is_same_code = False, is_all_zero = True,
                                                                                              normalize_input = normalize_input)

    map_ll_non_bursty4, rnn_ll_non_bursty4, map_ll_bursty4, rnn_ll_bursty4 = interpret.likelihood(bit_pos_list=[radar_bit_pos],sigma = 5.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True,
                                                                                              is_same_code = False, is_all_zero = True,
                                                                                              normalize_input = normalize_input)

    plt.figure(1)
    plt.subplot(121)
    plt.title('likelihood compare on different snr, RNN, Only AWGN Noise')
    p1, = plt.plot(rnn_ll_non_bursty1, 'y', label ='RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_non_bursty2, 'g', label ='RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_non_bursty3, 'b', label ='RNN AWGNsigma 2.0')
    p4, = plt.plot(rnn_ll_non_bursty4, 'r', label ='RNN AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])

    plt.subplot(122)
    plt.title('likelihood compare on different snr, BCJR/MAP, Only AWGN Noise')
    p1, = plt.plot(map_ll_non_bursty1, 'y', label ='BCJR AWGN sigma 0.5' )
    p2, = plt.plot(map_ll_non_bursty2, 'g', label ='BCJR AWGN sigma 1.0')
    p3, = plt.plot(map_ll_non_bursty3, 'b', label ='BCJR AWGNsigma 2.0')
    p4, = plt.plot(map_ll_non_bursty4, 'r', label ='BCJR AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])
    plt.show()

    plt.figure(2)
    plt.subplot(121)
    plt.title('likelihood compare on different snr, RNN, Radar Noise')
    p1, = plt.plot(rnn_ll_bursty1, 'y', label ='RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_bursty2, 'g', label ='RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_bursty3, 'b', label ='RNN AWGNsigma 2.0')
    p4, = plt.plot(rnn_ll_bursty4, 'r', label ='RNN AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])

    plt.subplot(122)
    plt.title('likelihood compare on different snr, BCJR/MAP, Radar Noise')
    p1, = plt.plot(map_ll_bursty1, 'y', label ='BCJR AWGN sigma 0.5' )
    p2, = plt.plot(map_ll_bursty2, 'g', label ='BCJR AWGN sigma 1.0')
    p3, = plt.plot(map_ll_bursty3, 'b', label ='BCJR AWGNsigma 2.0')
    p4, = plt.plot(map_ll_bursty4, 'r', label ='BCJR AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])
    plt.show()

def likelihood_1():
    ###############################################
    # Input Parameters
    ###############################################
    label1 = 'Hyeji Trained No Batch Norm'
    label2 = 'radar trained '
    label3 = 'awgn trained'

    network_saved_path_1 = './model_zoo/nobn_awgn/test1.h5'
    network_saved_path_2 = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_2.h5'
    network_saved_path_3 = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    radar_bit_pos = 50
    num_block = 100
    sigma_set = 1.0

    interpret_1  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block, rnn_type = 'gru', no_bn=True)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = sigma_set,
                                                                  radar_noise_power = 10, is_compute_map=True,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    K.clear_session()
    interpret_2 = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)

    rnn_ll_non_bursty6, rnn_ll_bursty6 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = sigma_set,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)



    K.clear_session()
    interpret_3 = Interpret(network_saved_path=network_saved_path_3, block_len=100, num_block=num_block)

    rnn_ll_non_bursty9, rnn_ll_bursty9 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = sigma_set,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    plt.figure(1)
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_non_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_non_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    #p3, = plt.plot(rnn_ll_non_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_non_bursty5, 'g', label =label2 +'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_non_bursty6], 'g-*', label =label2 +'RNN AWGN sigma'+ str(sigma_set))
    #p7, = plt.plot(rnn_ll_non_bursty7, 'g--', label =label2 +'RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_non_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_non_bursty9], 'b-*', label =label3 +'RNN AWGN sigma'+ str(sigma_set))
    #p0, = plt.plot(rnn_ll_non_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])

    plt.figure(2)
    plt.yscale('log')
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_non_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_non_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    #p3, = plt.plot(rnn_ll_non_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_non_bursty5, 'g', label =label2 +'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_non_bursty6], 'g-*', label =label2 +'RNN AWGN sigma'+ str(sigma_set))
    #p7, = plt.plot(rnn_ll_non_bursty7, 'g--', label =label2 +'RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_non_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_non_bursty9], 'b-*', label =label3 +'RNN AWGN sigma'+ str(sigma_set))
    #p0, = plt.plot(rnn_ll_non_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])

    plt.figure(3)
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    #p3, = plt.plot(rnn_ll_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_bursty5, 'g', label =label2 + 'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'g-*', label =label2 + 'RNN AWGN sigma'+ str(sigma_set))
    #p7, = plt.plot(rnn_ll_bursty7, 'g--', label =label2 + ' RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'b-*', label =label3 +'RNN AWGN sigma'+ str(sigma_set))
    #p0, = plt.plot(rnn_ll_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])

    plt.figure(4)
    plt.yscale('log')
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    #p3, = plt.plot(rnn_ll_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_bursty5, 'g', label =label2 + 'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'g-*', label =label2 + 'RNN AWGN sigma'+ str(sigma_set))
    #p7, = plt.plot(rnn_ll_bursty7, 'g--', label =label2 + ' RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'b-*', label =label3 +'RNN AWGN sigma'+ str(sigma_set))
    #p0, = plt.plot(rnn_ll_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])


    plt.figure(5)
    plt.title('likelihood compare on different snr at sigma ='+str(sigma_set)+'with Bursty Noise\n'+ label1+label2+label3)
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'g-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'k-*', label =label2 + 'RNN AWGN sigma'+ str(sigma_set))
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'y-*', label =label3 +'RNN AWGN sigma'+ str(sigma_set))

    p1, = plt.plot([abs(item) for item in map_ll_bursty2], 'r', label ='Turbo Bursty AWGN sigma'+ str(sigma_set))
    p0, = plt.plot([abs(item) for item in map_ll_non_bursty2], 'b', label ='Turbo non-bursty RNN AWGN sigma'+ str(sigma_set))

    plt.legend(handles = [p0, p1, p2, p6, p9])

    plt.figure(6)
    plt.yscale('log')
    plt.title('likelihood compare on different snr at sigma ='+str(sigma_set)+'with Bursty Noise\n'+ label1+label2+label3)
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'g-*', label =label1 + 'RNN AWGN sigma'+ str(sigma_set))
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'k-*', label =label2 + 'RNN AWGN sigma '+ str(sigma_set))
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'y-*', label =label3 +'RNN AWGN sigma '+ str(sigma_set))

    p1, = plt.plot([abs(item) for item in map_ll_bursty2], 'r', label ='Turbo Bursty AWGN sigma'+ str(sigma_set))
    p0, = plt.plot([abs(item) for item in map_ll_non_bursty2], 'b', label ='Turbo non-bursty RNN AWGN sigma'+ str(sigma_set))

    plt.legend(handles = [p0, p1, p2, p6, p9])

    plt.show()

def likelihood_radartrained_vs_awgntrained():
    ###############################################
    # Input Parameters
    ###############################################
    #label1 = 't-dist v3 trained '
    #label1 = 'Hyeji No BN GRU'
    label1 = 'Model 2 Trained'
    label2 = 'radar trained '
    label3 = 'awgn trained'

    #network_saved_path_1 = './model_zoo/nobn_awgn/test1.h5'
    network_saved_path_1 = './model_zoo/radar_model_end2end/hyeji_model_1004_trained_neg1_power20.h5'
    network_saved_path_2 = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_2.h5'
    network_saved_path_3 = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    radar_bit_pos = 50
    num_block = 200


    interpret_1  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block, rnn_type='lstm', no_bn=False)

    rnn_ll_non_bursty1, rnn_ll_bursty1 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=True,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)
    rnn_ll_non_bursty3, rnn_ll_bursty3 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    K.clear_session()
    interpret_2 = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)

    rnn_ll_non_bursty5, rnn_ll_bursty5 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty6, rnn_ll_bursty6 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty7, rnn_ll_bursty7 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    K.clear_session()
    interpret_3 = Interpret(network_saved_path=network_saved_path_3, block_len=100, num_block=num_block)

    rnn_ll_non_bursty8, rnn_ll_bursty8 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty9, rnn_ll_bursty9 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty0, rnn_ll_bursty0 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    plt.figure(1)
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_non_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_non_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_non_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_non_bursty5, 'g', label =label2 +'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_non_bursty6], 'g-*', label =label2 +'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_non_bursty7, 'g--', label =label2 +'RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_non_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_non_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_non_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])

    plt.figure(2)
    plt.yscale('log')
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_non_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_non_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_non_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_non_bursty5, 'g', label =label2 +'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_non_bursty6], 'g-*', label =label2 +'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_non_bursty7, 'g--', label =label2 +'RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_non_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_non_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_non_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])

    plt.figure(3)
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_bursty5, 'g', label =label2 + 'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'g-*', label =label2 + 'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_bursty7, 'g--', label =label2 + ' RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])

    plt.figure(4)
    plt.yscale('log')
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_bursty5, 'g', label =label2 + 'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'g-*', label =label2 + 'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_bursty7, 'g--', label =label2 + ' RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])


    plt.figure(5)
    plt.title('likelihood compare on different snr at sigma = 1.0 with Bursty Noise\n'+ label1+label2+label3)
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'g-*', label =label1 + 'RNN AWGN sigma 1.0')
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'k-*', label =label2 + 'RNN AWGN sigma 1.0')
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'y-*', label =label3 +'RNN AWGN sigma 1.0')

    p1, = plt.plot([abs(item) for item in map_ll_bursty2], 'r', label ='Turbo Bursty AWGN sigma 1.0')
    p0, = plt.plot([abs(item) for item in map_ll_non_bursty2], 'b', label ='Turbo non-bursty RNN AWGN sigma 1.0')

    plt.legend(handles = [p0, p1, p2, p6, p9])

    plt.figure(6)
    plt.yscale('log')
    plt.title('likelihood compare on different snr at sigma = 1.0 with Bursty Noise\n'+ label1+label2+label3)
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'g-*', label =label1 + 'RNN AWGN sigma 1.0')
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'k-*', label =label2 + 'RNN AWGN sigma 1.0')
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'y-*', label =label3 +'RNN AWGN sigma 1.0')

    p1, = plt.plot([abs(item) for item in map_ll_bursty2], 'r', label ='Turbo Bursty AWGN sigma 1.0')
    p0, = plt.plot([abs(item) for item in map_ll_non_bursty2], 'b', label ='Turbo non-bursty RNN AWGN sigma 1.0')

    plt.legend(handles = [p0, p1, p2, p6, p9])

    plt.show()


def likelihood_model_compare():
    ###############################################
    # Input Parameters
    ###############################################
    label1 = 't-dist v3 trained '
    label2 = 'model 2 trained '
    label3 = 'awgn trained'

    network_saved_path_1 = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_4.h5'
    #network_saved_path_2 = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_2.h5'
    network_saved_path_2 = './model_zoo/radar_model_end2end/hyeji_model_1004_trained_neg1_power20.h5'
    network_saved_path_3 = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    radar_bit_pos = 50
    num_block = 1000

    interpret_1  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block)

    rnn_ll_non_bursty1, rnn_ll_bursty1 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=True,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)
    rnn_ll_non_bursty3, rnn_ll_bursty3 = interpret_1.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    K.clear_session()
    interpret_2 = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)

    rnn_ll_non_bursty5, rnn_ll_bursty5 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty6, rnn_ll_bursty6 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty7, rnn_ll_bursty7 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    K.clear_session()
    interpret_3 = Interpret(network_saved_path=network_saved_path_3, block_len=100, num_block=num_block)

    rnn_ll_non_bursty8, rnn_ll_bursty8 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty9, rnn_ll_bursty9 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    rnn_ll_non_bursty0, rnn_ll_bursty0 = interpret_3.likelihood(bit_pos_list=[radar_bit_pos],sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False)

    plt.figure(1)
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_non_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_non_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_non_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_non_bursty5, 'g', label =label2 +'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_non_bursty6], 'g-*', label =label2 +'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_non_bursty7, 'g--', label =label2 +'RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_non_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_non_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_non_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])

    plt.figure(2)
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise\n'+ label1 +'vs' +label2)
    #p1, = plt.plot(rnn_ll_bursty1, 'y', label =label1 + 'RNN AWGN sigma 0.5' )
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'y-*', label =label1 + 'RNN AWGN sigma 1.0')
    #p3, = plt.plot(rnn_ll_bursty3, 'y--', label =label1 + 'RNN AWGNsigma 2.0')

    #p5, = plt.plot(rnn_ll_bursty5, 'g', label =label2 + 'RNN AWGN sigma 0.5' )
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'g-*', label =label2 + 'RNN AWGN sigma 1.0')
    #p7, = plt.plot(rnn_ll_bursty7, 'g--', label =label2 + ' RNN AWGNsigma 2.0')

    #p8, = plt.plot(rnn_ll_bursty8, 'b', label =label3 +'RNN AWGN sigma 0.5' )
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'b-*', label =label3 +'RNN AWGN sigma 1.0')
    #p0, = plt.plot(rnn_ll_bursty0, 'b--', label =label3 +'RNN AWGNsigma 2.0')
    plt.legend(handles = [p2, p6, p9])
    #plt.legend(handles = [p1, p2, p3, p5, p6, p7, p8, p9, p0])




    plt.figure(3)
    plt.title('likelihood compare on different snr at sigma = 1.0 with Bursty Noise\n'+ label1+label2+label3)
    p2, = plt.plot([abs(item) for item in rnn_ll_bursty2], 'g-*', label =label1 + 'RNN AWGN sigma 1.0')
    p6, = plt.plot([abs(item) for item in rnn_ll_bursty6], 'k-*', label =label2 + 'RNN AWGN sigma 1.0')
    p9, = plt.plot([abs(item) for item in rnn_ll_bursty9], 'y-*', label =label3 +'RNN AWGN sigma 1.0')

    p1, = plt.plot([abs(item) for item in map_ll_bursty2], 'r', label ='Turbo Bursty AWGN sigma 1.0')
    p0, = plt.plot([abs(item) for item in map_ll_non_bursty2], 'b', label ='Turbo non-bursty RNN AWGN sigma 1.0')

    plt.legend(handles = [p0, p1, p2, p6, p9])

    plt.show()

def ber_rnn_compare():
    print '[Interpret] Comparing between different RNN models'
    ###############################################
    # Input Parameters
    ###############################################
    #label1 = 't-dist v3 trained '
    label1 = 'model 1 trained'
    label2 = 'model 2 trained '
    label3 = 'awgn trained'

    #network_saved_path_1 = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_4.h5'
    network_saved_path_1 = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_2.h5'
    network_saved_path_2 = './model_zoo/radar_model_end2end/hyeji_model_1004_trained_neg1_power20.h5'
    network_saved_path_3 = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    radar_bit_pos = 50
    num_block = 1000


    # interpret_0  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block)
    # map_ber_non_bursty1, rnn_ber_non_bursty1, map_ber_bursty1, rnn_ber_bursty1 = interpret_0.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
    #                                                                           radar_noise_power = 10, is_compute_map=True,
    #                                                                           is_compute_no_bursty=True)

    interpret_1  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block)
    map_ber_non_bursty1, rnn_ber_non_bursty1, map_ber_bursty1, rnn_ber_bursty1 = interpret_1.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                              is_compute_no_bursty=True)
    interpret_2  = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)
    rnn_ber_non_bursty2,  rnn_ber_bursty2 = interpret_2.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=False,
                                                                              is_compute_no_bursty=True)
    interpret_3  = Interpret(network_saved_path=network_saved_path_3, block_len=100, num_block=num_block)
    rnn_ber_non_bursty3,  rnn_ber_bursty3 = interpret_3.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=False,
                                                                              is_compute_no_bursty=True)

    plt.figure(1)
    plt.title('Compare BER between RNN/Turbo\n over Turbo' + label1+label2+label3 + 'RNN')
    plt.xlabel('Position')
    p1, = plt.plot(map_ber_non_bursty1, 'b--', label ='Turbo Non Bursty' )
    p2, = plt.plot(rnn_ber_non_bursty1, 'g', label =label1 + 'RNN Non Bursty')
    p3, = plt.plot(map_ber_bursty1,     'b', label ='Turbo Bursty')
    p4, = plt.plot(rnn_ber_bursty1,     'g--', label =label1 + 'RNN Bursty')

    p5, = plt.plot(rnn_ber_non_bursty2, 'y--', label =label2 + 'RNN Non Bursty')
    p6, = plt.plot(rnn_ber_bursty2,     'y', label =label2 + 'RNN Bursty')
    p7, = plt.plot(rnn_ber_non_bursty3, 'k--', label =label3 + 'RNN Non Bursty')
    p8, = plt.plot(rnn_ber_bursty3,     'k', label =label3 + 'RNN Bursty')

    plt.legend(handles = [p1, p3,p4, p2, p5, p6, p7, p8])
    plt.show()

    plt.figure(2)
    plt.xlabel('Position')
    plt.title('Compare BER between RNN/Turbo\n over Turbo' + label1+label2+label3 + 'RNN')
    plt.yscale('log')
    p1, = plt.plot(map_ber_non_bursty1, 'b-*', label ='Turbo Non Bursty' )
    p2, = plt.plot(rnn_ber_non_bursty1, 'g-*', label =label1 + 'RNN Non Bursty')
    p3, = plt.plot(map_ber_bursty1,     'b', label ='Turbo Bursty')
    p4, = plt.plot(rnn_ber_bursty1,     'g', label =label1 + 'RNN Bursty')

    p5, = plt.plot(rnn_ber_non_bursty2, 'y-*', label =label2 + 'RNN Non Bursty')
    p6, = plt.plot(rnn_ber_bursty2,     'y', label =label2 + 'RNN Bursty')
    p7, = plt.plot(rnn_ber_non_bursty3, 'k-*', label =label3 + 'RNN Non Bursty')
    p8, = plt.plot(rnn_ber_bursty3,     'k', label =label3 + 'RNN Bursty')

    plt.legend(handles = [p1, p3,p4, p2, p5, p6, p7, p8])
    plt.show()

def ber_snr_range():
    print '[Interpret] BER output of Stacked RNN/ Turbo Decoder'
    ###############################################
    # Input Parameters
    ###############################################
    network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
    #network_saved_path = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_1.h5'
    interpret = Interpret(network_saved_path=network_saved_path, block_len=100, num_block=500, is_ll=False)

    radar_bit_pos = 50

    map_ber_non_bursty, rnn_ber_non_bursty, map_ber_bursty, rnn_ber_bursty = interpret.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                              is_compute_no_bursty=True)

    plt.figure(1)
    plt.title('Compare BER between RNN/Turbo')
    plt.yscale('log')
    p1, = plt.plot(map_ber_non_bursty, 'y', label ='Turbo Non Bursty' )
    p2, = plt.plot(rnn_ber_non_bursty, 'g', label ='RNN Non Bursty')
    p3, = plt.plot(map_ber_bursty,     'b', label ='Turbo Bursty')
    p4, = plt.plot(rnn_ber_bursty,     'k', label ='RNN Bursty')
    #plt.legend(handles = [p1, p3])
    plt.grid()
    plt.legend(handles = [p1, p2, p3, p4])
    plt.show()


def ber_bursty_only_compare():
    print '[Interpret] Comparing between different RNN models'
    ###############################################
    # Input Parameters
    ###############################################
    #label1 = 't-dist v3 trained '
    label1 = 'model 1 trained'
    label2 = 'model 2 trained'
    label3 = 'awgn trained'

    #network_saved_path_1 = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_4.h5'
    network_saved_path_1 = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_2.h5'
    network_saved_path_2 = './model_zoo/radar_model_end2end/hyeji_model_1004_trained_neg1_power20.h5'
    network_saved_path_3 = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'

    radar_bit_pos = 50
    num_block = 100


    # interpret_0  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block)
    # map_ber_non_bursty1, rnn_ber_non_bursty1, map_ber_bursty1, rnn_ber_bursty1 = interpret_0.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
    #                                                                           radar_noise_power = 10, is_compute_map=True,
    #                                                                           is_compute_no_bursty=True)

    interpret_1  = Interpret(network_saved_path=network_saved_path_1, block_len=100, num_block=num_block)
    map_ber_non_bursty1, rnn_ber_non_bursty1, map_ber_bursty1, rnn_ber_bursty1 = interpret_1.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                              is_compute_no_bursty=True)
    interpret_2  = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)
    rnn_ber_non_bursty2,  rnn_ber_bursty2 = interpret_2.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=False,
                                                                              is_compute_no_bursty=True)
    interpret_3  = Interpret(network_saved_path=network_saved_path_3, block_len=100, num_block=num_block)
    rnn_ber_non_bursty3,  rnn_ber_bursty3 = interpret_3.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=False,
                                                                              is_compute_no_bursty=True)

    plt.figure(1)
    plt.title('Compare BER between RNN/Turbo\n over Turbo' + label1+label2+label3 + 'RNN')
    plt.xlabel('Position')
    p1, = plt.plot(map_ber_non_bursty1, 'b--', label ='Turbo Non Bursty' )
    p2, = plt.plot(rnn_ber_non_bursty1, 'g', label =label1 + 'RNN Non Bursty')
    p3, = plt.plot(map_ber_bursty1,     'b', label ='Turbo Bursty')
    p4, = plt.plot(rnn_ber_bursty1,     'g--', label =label1 + 'RNN Bursty')

    p5, = plt.plot(rnn_ber_non_bursty2, 'y--', label =label2 + 'RNN Non Bursty')
    p6, = plt.plot(rnn_ber_bursty2,     'y', label =label2 + 'RNN Bursty')
    p7, = plt.plot(rnn_ber_non_bursty3, 'k--', label =label3 + 'RNN Non Bursty')
    p8, = plt.plot(rnn_ber_bursty3,     'k', label =label3 + 'RNN Bursty')

    plt.legend(handles = [p1, p3,p4, p2, p5, p6, p7, p8])
    plt.show()

    plt.figure(2)
    plt.xlabel('Position')
    plt.title('Compare BER between RNN/Turbo\n over Turbo' + label1+label2+label3 + 'RNN')
    plt.yscale('log')
    p1, = plt.plot(map_ber_non_bursty1, 'b-*', label ='Turbo Non Bursty' )
    p2, = plt.plot(rnn_ber_non_bursty1, 'g-*', label =label1 + 'RNN Non Bursty')
    p3, = plt.plot(map_ber_bursty1,     'b', label ='Turbo Bursty')
    p4, = plt.plot(rnn_ber_bursty1,     'g', label =label1 + 'RNN Bursty')

    p5, = plt.plot(rnn_ber_non_bursty2, 'y-*', label =label2 + 'RNN Non Bursty')
    p6, = plt.plot(rnn_ber_bursty2,     'y', label =label2 + 'RNN Bursty')
    p7, = plt.plot(rnn_ber_non_bursty3, 'k-*', label =label3 + 'RNN Non Bursty')
    p8, = plt.plot(rnn_ber_bursty3,     'k', label =label3 + 'RNN Bursty')

    plt.legend(handles = [p1, p3,p4, p2, p5, p6, p7, p8])
    plt.show()

def t_dist_for_paper():
    print '[Interpret] Likelihood output for T-dist'
    ###############################################
    # Input Parameters
    ###############################################


    network_saved_path_2 = './model_zoo/awgn_len100_end2end/awgn_bl100_1014.h5'

    radar_bit_pos = 50
    num_block = 5000

    interpret_2 = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)

    map_ll_non_bursty2_t, rnn_ll_non_bursty2_t, map_ll_bursty2_t, rnn_ll_bursty2_t = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=True,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False, is_t = True)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret_2.likelihood(bit_pos_list=[radar_bit_pos],sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=True,
                                                                  is_compute_no_bursty=True,
                                                                  is_same_code = True, is_all_zero = False, is_t = False)


    map_ll_non_bursty2_t = [abs(item) for item in map_ll_non_bursty2_t]
    rnn_ll_non_bursty2_t = [abs(item) for item in rnn_ll_non_bursty2_t]
    map_ll_non_bursty2 = [abs(item) for item in map_ll_non_bursty2]
    rnn_ll_non_bursty2 = [abs(item) for item in rnn_ll_non_bursty2]


    plt.figure(1)
    plt.title('Positional likelihood compare on different snr at sigma = 1.0 with t-dist noise')
    plt.ylabel('abs of likelihood')
    plt.xlabel('code block position')

    p1, = plt.plot(map_ll_non_bursty2_t, 'k', label ='Map T sigma 1.0')
    p2, = plt.plot(rnn_ll_non_bursty2_t, 'y', label ='RNN T sigma 1.0')
    p3, = plt.plot(map_ll_non_bursty2, 'k-*', label ='Map AWGN sigma 1.0')
    p4, = plt.plot(rnn_ll_non_bursty2, 'y-*', label ='RNN AWGN sigma 1.0')
    plt.grid()
    plt.legend(handles = [p1, p2, p3, p4])

    # plt.figure(2)
    # plt.title('likelihood compare on different snr at sigma = 1.0 with t-dist noise with all zero')
    # p1, = plt.plot(map_ll_bursty2_t, 'k', label ='Map T sigma 1.0')
    # p2, = plt.plot(rnn_ll_bursty2_t, 'y', label ='RNN T sigma 1.0')
    # p3, = plt.plot(map_ll_bursty2, 'k-*', label ='Map AWGN sigma 1.0')
    # p4, = plt.plot(rnn_ll_bursty2, 'y-*', label ='RNN AWGN sigma 1.0')
    # plt.grid()
    # plt.legend(handles = [p1, p2, p3, p4])



    plt.show()

def t_ber_for_paper():
    print '[Interpret] Comparing between different RNN models'
    ###############################################
    # Input Parameters
    ###############################################
    network_saved_path_2 = './model_zoo/awgn_len100_end2end/awgn_bl100_1014.h5'

    radar_bit_pos = 50
    num_block = 50000

    print 'num_block', num_block
    '1.18850222744'

    interpret_2  = Interpret(network_saved_path=network_saved_path_2, block_len=100, num_block=num_block)
    map_ber_non_bursty_t, rnn_ber_non_bursty_t = interpret_2.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                              is_compute_no_bursty=True,is_compute_bursty = False ,
                                                                              is_t=True)

    map_ber_non_bursty, rnn_ber_non_bursty    = interpret_2.ber(bit_pos_list=[radar_bit_pos], sigma=1.0,
                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                              is_compute_no_bursty=True,is_compute_bursty = False ,
                                                                              is_t=False)


    print 'map_ber_non_bursty_t', map_ber_non_bursty_t
    print 'rnn_ber_non_bursty_t', rnn_ber_non_bursty_t
    print 'map_ber_non_bursty',map_ber_non_bursty
    print 'rnn_ber_non_bursty',rnn_ber_non_bursty

    plt.figure(1)
    plt.title('Compare BER non bursty on T-dist and AWGN at -1.5dB')
    plt.xlabel('Position')
    p1, = plt.plot(map_ber_non_bursty_t, 'b--', label ='Turbo T dist' )
    p2, = plt.plot(rnn_ber_non_bursty_t, 'g--', label =  'RNN T dist')
    p3, = plt.plot(map_ber_non_bursty,     'b', label ='Turbo AWGN')
    p4, = plt.plot(rnn_ber_non_bursty,     'g', label ='RNN AWGN')

    plt.legend(handles = [p1,p2,  p3,p4])
    plt.show()

    # plt.figure(2)
    # plt.xlabel('Position')
    # plt.title('Compare BER between RNN/Turbo\n over Turbo' + label1+label2+label3 + 'RNN')
    # plt.yscale('log')
    # p1, = plt.plot(map_ber_non_bursty1, 'b-*', label ='Turbo Non Bursty' )
    # p2, = plt.plot(rnn_ber_non_bursty1, 'g-*', label =label1 + 'RNN Non Bursty')
    # p3, = plt.plot(map_ber_bursty1,     'b', label ='Turbo Bursty')
    # p4, = plt.plot(rnn_ber_bursty1,     'g', label =label1 + 'RNN Bursty')
    #
    # p5, = plt.plot(rnn_ber_non_bursty2, 'y-*', label =label2 + 'RNN Non Bursty')
    # p6, = plt.plot(rnn_ber_bursty2,     'y', label =label2 + 'RNN Bursty')
    # p7, = plt.plot(rnn_ber_non_bursty3, 'k-*', label =label3 + 'RNN Non Bursty')
    # p8, = plt.plot(rnn_ber_bursty3,     'k', label =label3 + 'RNN Bursty')
    #
    # plt.legend(handles = [p1, p3,p4, p2, p5, p6, p7, p8])
    #plt.show()


if __name__ == '__main__':
    # User Case 1, likelihood on bursty noise and AWGN only. Compare RNN and BCJR's output.
    #likelihood_snr_range()

    # User Case 2, likelihood on RNN models. Compare the output scale.
    #likelihood_radartrained_vs_awgntrained()
    #likelihood_1()

    # User Case 3, ber on bursty noise. Compare stacked RNN decoder and Turbo Decoder's BER bit-wise
    #ber_snr_range()

    # User Case 4, ber on bursty noise over RNN models.
    #ber_rnn_compare()

    # User Case 5
    #likelihood_model_compare()

    # for paper
    #ber_bursty_only_compare()

    #likelihood_snr_range()


    t_dist_for_paper()
    #t_ber_for_paper()










