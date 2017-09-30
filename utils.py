__author__ = 'yihanjiang'
'''
This is util functions of the decoder
SHall involve:
(1) Adding noise
(2) Parallel Generating Turbo Code?
'''

import numpy as np
import math
import commpy.channelcoding.turbo as turbo

from scipy import stats
#######################################
# Interleaving Helper Functions
#######################################
def deint(in_array, p_array):
    out_array = np.zeros(len(in_array), in_array.dtype)
    for index, element in enumerate(p_array):
        out_array[element] = in_array[index]
    return out_array

def intleave(in_array, p_array):
    out_array = np.array(map(lambda x: in_array[x], p_array))
    return out_array

def direct_subtract(in1,in2):
    out = in1
    out[:,:,2] = in1[:,:,2] - in2
    return out

#######################################
# Noise Helper Function
#######################################
'''
Use Corrupt_signal, generate_noise is depreciated.
'''
def corrupt_signal(input_signal, noise_type, sigma = 1.0,
                    vv =5.0, radar_power = 20.0, radar_prob = 5e-2, denoise_thd = 10.0,
                    modulate_mode = 'bpsk'):
    '''
    Documentation TBD.
    only bpsk is allowed, but other modulation is allowed by user-specified modulation.
    :param noise_type: required, choose from 'awgn', 't-dist'
    :param sigma:
    :param data_shape:
    :param vv: parameter for t-distribution.
    :param radar_power:
    :param radar_prob:
    :return:
    '''

    data_shape = input_signal.shape  # input_signal has to be a numpy array.

    if noise_type == 'awgn':
        noise = sigma * np.random.standard_normal(data_shape) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 't-dist':
        noise = sigma * math.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'awgn+radar':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        add_poscomp = np.ones(data_shape) - abs(add_pos)

        corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        # noise = sigma * np.random.standard_normal(data_shape) + \
        #         np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        #
        # corrupted_signal = 2.0*input_signal-1.0  + noise

    elif noise_type == 'radar':
        noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'awgn+radar+denoise':
        bpsk_signal = 2.0*input_signal-1.0 + sigma * np.random.standard_normal(data_shape)
        add_pos     = np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        add_poscomp = np.ones(data_shape) - abs(add_pos)
        corrupted_signal = bpsk_signal * add_poscomp + np.random.normal(radar_power, 1.0,size = data_shape ) * add_pos

        corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

        # noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])
        # corrupted_signal = 2.0*input_signal-1.0 + noise
        # corrupted_signal  = stats.threshold(corrupted_signal, threshmin=-denoise_thd, threshmax=denoise_thd, newval=0.0)

    else:
        noise = sigma * np.random.standard_normal(data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    return corrupted_signal

def generate_noise(noise_type, sigma, data_shape, vv =5.0, radar_power = 20.0, radar_prob = 5e-2):
    '''
    Documentation TBD.
    :param noise_type: required, choose from 'awgn', 't-dist'
    :param sigma:
    :param data_shape:
    :param vv: parameter for t-distribution.
    :param radar_power:
    :param radar_prob:
    :return:
    '''
    if noise_type == 'awgn':
        noise = sigma * np.random.standard_normal(data_shape) # Define noise

    elif noise_type == 't-dist':
        noise = sigma * math.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)

    elif noise_type == 'awgn+radar':
        noise = sigma * np.random.standard_normal(data_shape) + \
                np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])

    elif noise_type == 'radar':
        noise = np.random.normal(radar_power, 1.0,size = data_shape ) * np.random.choice([-1.0, 0.0, 1.0], data_shape, p=[radar_prob/2, 1 - radar_prob, radar_prob/2])

    else:
        noise = sigma * np.random.standard_normal(data_shape)

    return noise

#######################################
# Build RNN Feed Helper Function
#######################################

def build_rnn_data_feed(num_block, block_len, noiser, codec, is_all_zero = False ,is_same_code = False, **kwargs):
    '''

    :param num_block:
    :param block_len:
    :param noiser: list, 0:noise_type, 1:sigma,     2:v for t-dist, 3:radar_power, 4:radar_prob
    :param codec:  list, 0:trellis1,   1:trellis2 , 2:interleaver
    :param kwargs:
    :return: X_feed, X_message
    '''

    # Unpack Noiser
    noise_type  = noiser[0]
    noise_sigma = noiser[1]
    vv          = 5.0
    radar_power = 20.0
    radar_prob  = 5e-2
    denoise_thd = 10.0

    if noise_type == 't-dist':
        vv = noiser[2]
    elif noise_type == 'awgn+radar':
        radar_power = noiser[3]
        radar_prob  = noiser[4]

    elif noise_type == 'awgn+radar+denoise':
        radar_power = noiser[3]
        radar_prob  = noiser[4]
        denoise_thd = noiser[5]
    elif noise_type == 'customize':
        '''
        TBD, noise model shall be open to other user, for them to train their own decoder.
        '''

        print '[Debug] Customize noise model not supported yet'
    else:  # awgn
        pass

    # Unpack Codec
    trellis1    = codec[0]
    trellis2    = codec[1]
    interleaver = codec[2]
    p_array     = interleaver.p_array

    X_feed = []
    X_message = []

    same_code = np.random.randint(0, 2, block_len)

    for nbb in range(num_block):
        if is_same_code:
            message_bits = same_code
        else:
            if is_all_zero == False:
                message_bits = np.random.randint(0, 2, block_len)
            else:
                message_bits = np.random.randint(0, 1, block_len)

        X_message.append(message_bits)
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        sys_r  = corrupt_signal(sys, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)
        par1_r = corrupt_signal(par1, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)
        par2_r = corrupt_signal(par2, noise_type =noise_type, sigma = noise_sigma,
                               vv =vv, radar_power = radar_power, radar_prob = radar_prob, denoise_thd = denoise_thd)


        rnn_feed_raw = np.stack([sys_r, par1_r, np.zeros(sys_r.shape), intleave(sys_r, p_array), par2_r], axis = 0).T
        rnn_feed = rnn_feed_raw

        X_feed.append(rnn_feed)

    X_feed = np.stack(X_feed, axis=0)

    X_message = np.array(X_message)
    X_message = X_message.reshape((-1,block_len, 1))

    return X_feed, X_message

#######################################
# Helper Function for convert SNR
#######################################

def snr_db2sigma(train_snr):
    block_len    = 1000
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def snr_sigma2db(sigma_snr):
    SNR          = -10*np.log10(sigma_snr**2)
    return SNR




if __name__ == '__main__':
    pass