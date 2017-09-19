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