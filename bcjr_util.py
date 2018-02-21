import time
import numpy as np
import math
import commpy.channelcoding.turbo as turbo
from utils import corrupt_signal, snr_db2sigma

def generate_bcjr_example(num_block, block_len, codec, num_iteration, is_save = True, train_snr_db = 0.0, save_path = './tmp/',
                          **kwargs ):
    '''
    Generate BCJR feature and target for training BCJR-like RNN codec from scratch
    '''

    start_time = time.time()
    # print
    print '[BCJR] Block Length is ', block_len
    print '[BCJR] Number of Block is ', num_block

    input_feature_num = 3
    noise_type  = 'awgn'
    noise_sigma = snr_db2sigma(train_snr_db)

    identity = str(np.random.random())    # random id for saving

    # Unpack Codec
    trellis1    = codec[0]
    trellis2    = codec[1]
    interleaver = codec[2]

    # Initialize BCJR input/output Pair for training (Is that necessary?)
    bcjr_inputs  = np.zeros([2*num_iteration, num_block, block_len ,input_feature_num])
    bcjr_outputs = np.zeros([2*num_iteration, num_block, block_len ,1        ])

    for block_idx in range(num_block):
        # Generate Noisy Input For Turbo Decoding
        message_bits = np.random.randint(0, 2, block_len)
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        sys_r  = corrupt_signal(sys, noise_type =noise_type, sigma = noise_sigma,)
        par1_r = corrupt_signal(par1, noise_type =noise_type, sigma = noise_sigma)
        par2_r = corrupt_signal(par2, noise_type =noise_type, sigma = noise_sigma)

        # Use the Commpy BCJR decoding algorithm
        sys_symbols = sys_r
        non_sys_symbols_1 = par1_r
        non_sys_symbols_2 = par2_r
        noise_variance = noise_sigma**2
        sys_symbols_i = interleaver.interlv(sys_symbols)
        trellis = trellis1

        L_int = None
        if L_int is None:
            L_int = np.zeros(len(sys_symbols))

        L_int_1 = L_int
        L_ext_2 = L_int_1

        weighted_sys = 2*sys_symbols*1.0/noise_variance # Is gonna be used in the final step of decoding.
        weighted_sys_int = interleaver.interlv(weighted_sys)

        for turbo_iteration_idx in range(num_iteration-1):
            L_int_1 = interleaver.deinterlv(L_ext_2)
            # MAP 1
            [L_ext_1, decoded_bits] = turbo.map_decode(sys_symbols, non_sys_symbols_1,
                                                 trellis, noise_variance, L_int_1, 'compute')
            L_ext_1 -= L_int_1
            L_ext_1 -= weighted_sys

             # ADD Training Examples
            bcjr_inputs[2*turbo_iteration_idx,block_idx,:,:] = np.concatenate([sys_symbols.reshape(block_len,1),
                                                                               non_sys_symbols_1.reshape(block_len,1),
                                                                               L_int_1.reshape(block_len,1)],
                                                                              axis=1)
            bcjr_outputs[2*turbo_iteration_idx,block_idx,:,:]= L_ext_1.reshape(block_len,1)

            # MAP 2
            L_int_2 = interleaver.interlv(L_ext_1)

            [L_ext_2, decoded_bits] = turbo.map_decode(sys_symbols_i, non_sys_symbols_2,
                                             trellis, noise_variance, L_int_2, 'compute')
            L_ext_2 -=  L_int_2
            L_ext_2 -=  weighted_sys_int
            # ADD Training Examples
            bcjr_inputs[2*turbo_iteration_idx+1,block_idx,:,:] = np.concatenate([sys_symbols_i.reshape(block_len,1),
                                                                                 non_sys_symbols_2.reshape(block_len,1),
                                                                                 L_int_2.reshape(block_len,1)],
                                                                                axis=1)
            bcjr_outputs[2*turbo_iteration_idx+1,block_idx,:,:] = L_ext_2.reshape(block_len,1)

        # MAP 1
        L_int_1 = interleaver.deinterlv(L_ext_2)
        [L_ext_1, decoded_bits] = turbo.map_decode(sys_symbols, non_sys_symbols_1,
                                             trellis, noise_variance, L_int_1, 'compute')
        L_ext_1 -= L_int_1
        L_ext_1 -= weighted_sys
         # ADD Training Examples


        bcjr_inputs[2*num_iteration-2,block_idx,:,:] = np.concatenate([sys_symbols.reshape(block_len,1),
                                                                     non_sys_symbols_1.reshape(block_len,1),
                                                                     L_int_1.reshape(block_len,1)],
                                                                    axis=1)
        bcjr_outputs[2*num_iteration-2,block_idx,:,:] = L_ext_1.reshape(block_len,1)

        # MAP 2
        L_int_2 = interleaver.interlv(L_ext_1)
        [L_2, decoded_bits] = turbo.map_decode(sys_symbols_i, non_sys_symbols_2,
                                         trellis, noise_variance, L_int_2, 'decode')
        L_ext_2 = L_2 - L_int_2
        L_ext_2 -=  weighted_sys_int
        # ADD Training Examples
        bcjr_inputs[2*num_iteration-1,block_idx,:,:] = np.concatenate([sys_symbols_i.reshape(block_len,1),
                                                                       non_sys_symbols_2.reshape(block_len,1),
                                                                       L_int_2.reshape(block_len,1)],
                                                                      axis=1)
        bcjr_outputs[2*num_iteration-1,block_idx,:,:] = L_ext_2.reshape(block_len,1)

    end_time = time.time()
    print '[BCJR] The input feature has shape', bcjr_inputs.shape,'the output has shape', bcjr_outputs.shape
    print '[BCJR] Generating Training Example takes ', end_time - start_time , 'secs'
    print '[BCJR] file id is', identity

    bcjr_inputs_train   = bcjr_inputs.reshape((-1, block_len,input_feature_num ))
    bcjr_outputs_train  = bcjr_outputs.reshape((-1,  block_len, 1))

    target_train_select = bcjr_outputs_train[:,:,0] + bcjr_inputs_train[:,:,2]

    target_train_select[:,:] = math.e**target_train_select[:,:]*1.0/(1+math.e**target_train_select[:,:])

    X_input  = bcjr_inputs_train.reshape(-1,block_len,input_feature_num)
    X_target = target_train_select.reshape(-1,block_len,1)

    return X_input, X_target


if __name__ == '__main__':

    import commpy.channelcoding.interleavers as RandInterlv
    import commpy.channelcoding.convcode as cc

    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[7, 5]])
    feedback = 7
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(100, 0)
    p_array = interleaver.p_array
    print '[Turbo Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback

    ##########################################
    # Setting Up RNN Model
    ##########################################
    codec  = [trellis1, trellis2, interleaver]

    generate_bcjr_example(num_block=10000, block_len=100, codec=codec, num_iteration=6)
