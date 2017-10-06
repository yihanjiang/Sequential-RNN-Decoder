########################################################
# evaluate_rnn.py
# First Commit: 09/19/2017 Yihan Jiang
########################################################
Functionality: Evaluate RNN Bit Error Rate (BER) and Block Error Rate (BLER).

Usage: python evaluate_rnn.py [arguments]

By default (with no arguments) it runs with AWGN trained LSTM Turbo Decoder. The final output is SNRs, BERs and BLERs.

Argements:
(1) -block_len:          Block Length For Turbo Code, by default is 1000.
(2) -num_block:          Number of Blocks. By default is 100, but for high SNR need to be larger.
(3) -num_dec_iteration:  Number of Decoding Iteration, by default is 6.
(4) -noise_type:         Default: 'awgn', you can choose between 'awgn', 't-dist', 'awgn+radar' and 'awgn+radar+denoise'
             -v:         Default v = 5.0, v is the parameter for t-distribution, which range from 2.0 to infinity.
    -radar_power:        Default 20 with var=1, only valid when noise_type is 'awgn+radar' and 'awgn+radar+denoise'.
    -radar_prob:         Default 0.05, only valid when noise_type is 'awgn+radar' and 'awgn+radar+denoise'.
    -denoise_thd:        Default 10, could change accordingly.
    
(5) -network_model_path: Default points to a AWGN trained network

(6) -num_hidden_unit:    Default 200. Has to match the network you are using
    -rnn_type:           Default lstm, choose from lstm, gru and simple-rnn
    -num_layer:          Default 2, should be consistent to the model you trained. Now only support 1 and 2.
    -rnn_direction:      Default bd, choose from 'sd' and 'bd'
    
(7) -snr_range:          Default -1 dB to 2 dB
(8) -snr_points:         Default 10
(9) -codec_type:         Default uses M=2, (7,5), feedback = 7. 'lte' uses LTE Turbo Codec. Codec has to match network model
(*) --help:              Show Help Page

Example:
(1) To evaluate BER/BLER of AWGN Channel of Turbo RNN Decoder, from -1 dB to 2 dB, 10 SNR points, with default setting.
python evaluate_rnn.py

(2) To evaluate BER/BLER of AWGN Channel of Turbo RNN Decoder with more blocks (Higher accuracy, avoid 0 in BER curve)
python evaluate_rnn.py -num_block 1000

(3) To evaluate BER/BLER of t-distribution Channel with v=3.0, with t-distribution trained network
python evaluate_rnn.py -num_block 1000 -noise_type t-dist -v 3.0 -network_model_path 'model_path'

(4) See Manual for evaluate_rnn.py
python evaluate_rnn.py --help

Warning: Running on over 1000 blocks could take over 24 hours, start by small number of blocks!
