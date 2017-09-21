Documentation for RNN Turbo Decoder
Author: Yihan Jiang, UW & Hyeji Kim, UIUC.

We have a few scripts for user.
(1) evaluate_performance.py is a command line tool for evaluating Turbo RNN Decoder's BER curve.
(2) train_turbo_decoder_end2end.py is a command line tool for training Turbo Decoder with different noise.
(3) train_bcjr_rnn.py is a command line
(4) TBD: Support User Defined Codec
(5) TBD: User Defined Channel, or let user override channel easily. 

Dependency:
(0) Python (2.7.10+)
(1) numpy (1.13.1)
(2) Keras (2.0.6)
(3) scikit-commpy (0.3.0) For Commpy, we use a modified version
                              of the original commpy, which is in the folder with name commpy.
                              Commpy will be depreciated for future versions.
(4) h5py (2.7.0)
(5) tensorflow (1.2.1)

Use pip to install above packages.

########################################################
# evaluate_performance
########################################################
Functionality: Evaluate RNN BER only.
Usage Example: python evaluate_performance.py [arguments]
By default (with no arguments) it runs with AWGN trained LSTM Turbo Decoder. The final output is SNRs, BERs and BLERs.

Argements:
(1) -block_len:          Block Length For Turbo Code, by default is 1000.
(2) -num_block:          Number of Blocks. By default is 100, but for high SNR need to be larger.
(3) -num_dec_iteration:  Number of Decoding Iteration, by default is 6.
(4) -noise_type:         Default: 'awgn', you can choose between 'awgn', 't-dist'
             -v:         Default v = 5.0, v is the parameter for t-distribution, which range from 2.0 to infinity.
   -radar_power:         Default 20 with var=1, only valid when noise_type is 'awgn+radar'.
    -radar_prob:         Default 0.05, only valid when noise_type is 'awgn+radar'.
(5) -network_model_path: Default trained with AWGN
(6) -num_dec_iteration:  (Not working now!)Default 200
(7) -snr_range:          Default -1 dB to 2 dB
(8) -snr_points:         Default 10
(*) --help:              Show Help Page


TBD Arguments (Not implemented):
(1) -network_type        'bd-lstm', 'sd-lstm', 'bd-gru', 'sd-gru'
(2) TBD

########################################################
# train_turbo_decoder_end2end.py
########################################################
Usage: Train RNN at specific SNR with specific channel setup.

(*) --help:              Show Help Page

(1) -block_len:          Block Length For Turbo Code, by default is 1000.
(2) -num_block_train:    Number of Train Blocks. By default is 100
    -num_block_test:     Number of Test Blocks. By default is 10, but for high SNR need to be larger.
(3) -num_dec_iteration:  Number of Decoding Iteration, by default is 6.
(4) -noise_type:         Default: 'awgn', you can choose between 'awgn', 'awgn+radar', 't-dist'
             -v:         Default v = 5.0, v is the parameter for t-distribution, which range from 2.0 to infinity.
    -radar_power:         Default 20 with var=1, only valid when noise_type is 'awgn+radar'.
    -radar_prob:         Default 0.05, only valid when noise_type is 'awgn+radar'.
(5) -network_model_path: Default trained with AWGN
(6) -num_dec_iteration:  (Not working now!)Default 12
(7) -train_snr:          Default -1dB,  -1,0,1dB recommended.
(8) -save_path:          Default saved at ./tmp/+some random number.h5
(9) -learning_rate:      Default 1e-3
(10)-batch_size:         Default 10
(11)


TBD Argument (Not Implemented):

(*) -optimizer:          Default Adam (not implemented yet)




