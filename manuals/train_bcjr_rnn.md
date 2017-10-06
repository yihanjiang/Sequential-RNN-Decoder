########################################################
# train_bcjr_rnn.py
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
(7) -snr_range:          Default -1 dB to 2 dB
(8) -snr_points:         Default 10
(9) -codec_type:         Default uses M=2, (7,5), feedback = 7. 'lte' uses LTE Turbo Codec. Codec has to match network model
(*) --help:              Show Help Page


Usage: Train BCJR-like RNN at specific SNR with specific channel setup.

bcjr_outputs, bcjr_inputs, num_iteration, block_len

(*) --generate_example   Decide generate example from scratch, or load existing training example. Use in front of the arguments
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
(11)-rnn_direction
(12)-rnn_type            Choose from 'rnn-gru','rnn-lstm' now



(13)-codec_type          'lte' or 'default'. Predefined Codec



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
