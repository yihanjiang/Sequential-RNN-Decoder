__author__ = 'yihanjiang'

import numpy as np
import math
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional

from keras import backend as K
from keras.engine import Layer

#######################################
# TBD: Customize Layer for future use
#######################################
# TBD
class TurboRNNLayer(Layer):

    # def __init__(self, output_dim, **kwargs):
    #     self.output_dim = output_dim
    #     super(MyLayer, self).__init__(**kwargs)
    #
    # def build(self, input_shape):
    #     # Create a trainable weight variable for this layer.
    #     self.kernel = self.add_weight(name='kernel',
    #                                   shape=(input_shape[1], self.output_dim),
    #                                   initializer='uniform',
    #                                   trainable=True)
    #     super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!
    #
    # def call(self, x):
    #     return K.dot(x, self.kernel)
    #
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.output_dim)

    def __init__(self, interleave_array, num_hidden_unit = 200, num_iteration = 6,  **kwargs):
        self.supports_masking = True
        self.interleave_array = interleave_array

        super(TurboRNNLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if len(inputs.get_shape()) != 3:
            raise ValueError('Aaaaa our input is not of dim 3!')
        a = inputs

        output_list = [None for _ in range(len(self.interleave_array))]
        b = []
        for step_index, element in enumerate(self.interleave_array):
            b.append(a[:,step_index,:])

        for origal_index, inter_index in zip(range(len(self.interleave_array)), self.interleave_array):
            output_list[inter_index] = b[origal_index]

        res = K.stack(output_list, axis=1)
        return res

    def get_output_shape_for(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_output_shape(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_mask(self, inputs, masks=None):
        if masks is None:
            return None
        return masks[1]

#######################################
# TBD: Customize Layer for future use
#######################################
# TBD

class DeInterleave(Layer):
    """
    Customized Layer for DeInterleaver, no Parameter to train
    Example:
    interleaver_array = [2,3,5,1,8,9,7,0,4,6]
    input = Input(shape = (10,2))

    x = Interleave(interleave_array=interleaver_array)(input)
    model = Model(inputs=input, outputs=x)
    optimizer= keras.optimizers.adam(lr=0.01)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=['accuracy'])

    fake_data = np.array([range(100)]).reshape((-1, 10, 2))
    inter_data =  model.predict(fake_data, batch_size=32)
    input = Input(shape = (10,2))

    x = DeInterleave(interleave_array=interleaver_array)(input)
    model = Model(inputs=input, outputs=x)
    optimizer= keras.optimizers.adam(lr=0.01)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=['accuracy'])

    deinter_data =  model.predict(inter_data, batch_size=32)
    print deinter_data[0]
    print fake_data[0]
    """

    def __init__(self, interleave_array, **kwargs):
        self.supports_masking = True
        self.interleave_array = interleave_array
        super(DeInterleave, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if len(inputs.get_shape()) != 3:
            raise ValueError('Aaaaa our input is not of dim 3!')
        a = inputs

        output_list = [None for _ in range(len(self.interleave_array))]
        b = []
        for step_index, element in enumerate(self.interleave_array):
            b.append(a[:,step_index,:])

        for origal_index, inter_index in zip(range(len(self.interleave_array)), self.interleave_array):
            output_list[inter_index] = b[origal_index]

        res = K.stack(output_list, axis=1)
        return res

    def get_output_shape_for(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_output_shape(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_mask(self, inputs, masks=None):
        if masks is None:
            return None
        return masks[1]

class Interleave(Layer):
    """
    Customized Layer for Interleaver, no Parameter to train
    """

    def __init__(self, interleave_array, **kwargs):
        self.supports_masking = True
        self.interleave_array = interleave_array
        super(Interleave, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if len(inputs.get_shape()) != 3:
            raise ValueError('Aaaaa our input is not of dim 3!')
        a = inputs
        # if K.ndim(a) != 3 or K.ndim(b) != 3:
        #     raise ValueError('Interleaved tensors must have ndim 3')
        output_list = []
        for step_index in self.interleave_array:
            this_index = step_index
            b = a[:,this_index,:]
            output_list.append(b)

        res = K.stack(output_list, axis=1)
        return res

    def get_output_shape_for(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_output_shape(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])

    def compute_mask(self, inputs, masks=None):
        if masks is None:
            return None
        return masks[1]

def load_model(interleave_array, dec_iter_num = 6,block_len = 1000,  network_saved_path='default', num_layer = 2,
               learning_rate = 0.001, num_hidden_unit = 200, rnn_type = 'lstm',rnn_direction = 'bd',
               last_layer_sigmoid = True, loss= 'mean_squared_error',
               **kwargs):

    if network_saved_path == 'default':
        network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
    else:
        network_saved_path = network_saved_path

    #rnn_type    = 'lstm'    #'gru', 'lstm'
    print '[RNN Model] using model type', rnn_type
    print '[RNN Model] using model path', network_saved_path
    ######################################
    # Encode Turbo Code
    ######################################
    batch_size    = 32

    print '[RNN Model] Block length', block_len
    print '[RNN Model] Evaluate Batch size', batch_size
    print '[RNN Model] Number of decoding layers', dec_iter_num

    def errors(y_true, y_pred):
        myOtherTensor = K.not_equal(y_true, K.round(y_pred))
        return K.mean(tf.cast(myOtherTensor, tf.float32))

    ####################################################
    # Define Model
    ####################################################
    if rnn_direction == 'bd':
        if rnn_type == 'lstm':
            f1 = Bidirectional(LSTM(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(LSTM(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')
        elif rnn_type == 'gru':
            f1 = Bidirectional(GRU(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(GRU(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')
        else: #SimpleRNN
            f1 = Bidirectional(SimpleRNN(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(SimpleRNN(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')

    elif rnn_direction == 'sd':
        if rnn_type == 'lstm':
            f1 = LSTM(name='lstm_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = LSTM(name='lstm_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f4 = BatchNormalization(name='batch_normalization_2')
        elif rnn_type == 'gru':
            f1 = GRU(name='gru_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = GRU(name='gru_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f4 = BatchNormalization(name='batch_normalization_2')
        else: #SimpleRNN
            f1 = SimpleRNN(name='simple_rnn_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = SimpleRNN(name='simple_rnn_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0)
            f4 = BatchNormalization(name='batch_normalization_2')
    else:
        print '[RNN Model]RNN direction not supported, exit'
        import sys
        sys.exit()

    f5 = TimeDistributed(Dense(1),name='time_distributed_1')

    if last_layer_sigmoid:
        f6 = TimeDistributed(Dense(1,activation='sigmoid'),name='time_distributed_sigmoid')
    else:
        f6 = TimeDistributed(Dense(1),name='time_distributed_sigmoid')

    inputs = Input(shape = (block_len,5))
    #interleave_array = interleaver.p_array
    interleave_array = interleave_array

    def split_data_0(x):
        x1 = x[:,:,0:3]
        return x1

    def split_data_1(x):
        x1 = x[:,:,0:2]
        return x1

    def split_data_2(x):
        xx = x[:,:,3:5]
        return xx

    def takeLL(x):
        #x1_out = x[:,:,0]
        return tf.reshape(x[:,:,0]-x[:,:,3],[tf.shape(x[:,:,0])[0],block_len,1])
        #return tf.reshape(x[:,:,0],[tf.shape(x)[0],block_len,1])
        #return x

    def concat(x):
        return K.concatenate(x)

    takell = Lambda(takeLL)
    lambda_concat = Lambda(concat)
    DeInt = DeInterleave(interleave_array=interleave_array)
    DoInt = Interleave(interleave_array=interleave_array)

    def subtr(x2):
        # x2_out = f5(f4(f3(f2(f1(x2)))))
        # return x2_out
        if num_layer == 2:
            x2_out = f5(f4(f3(f2(f1(x2)))))
        elif num_layer == 1:
            x2_out = f5(f2(f1(x2)))
        else:
            print 'other layer not supported!'
            return
        x2_temp = lambda_concat([x2_out, x2])
        x2 = takell(x2_temp)
        return x2

    # def subtr_sigmoid(x2):
    #     x2_out = f6(f4(f3(f2(f1(x2)))))
    #     x2_temp = Lambda(concat)([x2_out, x2])
    #     x2 = Lambda(takeLL)(x2_temp)
    #     return x2

    x_input_1 = Lambda(split_data_1, name = 'split_data_normal')(inputs) # sys, par1
    x_input_2 = Lambda(split_data_2, name = 'split_data_interleave')(inputs) # sys_i, par2

    x1 = Lambda(split_data_0, name='three')(inputs) # sys, par1, 0 (initial likelihood)
    x1 = subtr(x1)#x1 = f5(f4(f3(f2(f1(x1)))))
    x1 = DoInt(x1)

    x2 = lambda_concat([x_input_2, x1])
    x2 = subtr(x2)#x2 = f5(f4(f3(f2(f1(x2)))))
    x2 = DeInt(x2)

    for dec_iter in range(dec_iter_num-2):
        x3 = lambda_concat([x_input_1, x2])
        x3 = subtr(x3)#x3 = f5(f4(f3(f2(f1(x3)))))
        x3 = DoInt(x3)

        x4 = lambda_concat([x_input_2, x3])
        x4 = subtr(x4)
        x4 = DeInt(x4)
        x2 = x4

    x3 = lambda_concat([x_input_1, x2])
    x3 = subtr(x3)#x3 = f5(f4(f3(f2(f1(x3)))))
    x3 = DoInt(x3)

    x4 = lambda_concat([x_input_2, x3])

    if num_layer == 2:
        x4 = f6(f4(f3(f2(f1(x4)))))
    elif num_layer == 1:
        x4 = f6(f2(f1(x4)))

    x4 = DeInt(x4)

    predictions = x4

    model = Model(inputs=inputs, outputs=predictions)
    optimizer= keras.optimizers.adam(lr=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer,loss=loss, metrics=[errors])

    try:
        model.load_weights(network_saved_path, by_name=True)
    except:
        print '[RNN Model][Warning]loading weight fails!'

    #print model.summary()

    layer_from = model.get_layer('time_distributed_1')
    weights = layer_from.get_weights()
    layer_to = model.get_layer('time_distributed_sigmoid')
    layer_to.set_weights(weights)

    return model

def test_TurboLayer():
    '''
    Test If Turbo Layer works.
    :return:
    '''
    pass


if __name__ == '__main__':
    #load_model()

    test_TurboLayer()