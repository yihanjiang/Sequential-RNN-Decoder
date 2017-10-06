########################################################
# evaluate_commpy.py
# First Commit: 09/19/2017 Yihan Jiang
########################################################
Functionality: Evaluate Commpy (Turbo BCJR Decoder) Bit Error Rate (BER) and Block Error Rate (BLER).

Usage: python evaluate_commpy.py [arguments]

By default (with no arguments) it runs with AWGN Channel. The final output is SNRs, BERs and BLERs.

Argements:
(1) -block_len:          Block Length For Turbo Code, by default is 1000.
(2) -num_block:          Number of Blocks. By default is 100, but for high SNR need to be larger.
(3) -num_dec_iteration:  Number of Decoding Iteration, by default is 6.
(4) -noise_type:         Default: 'awgn', you can choose between 'awgn', 't-dist', 'awgn+radar' and 'awgn+radar+denoise'
             -v:         Default v = 5.0, v is the parameter for t-distribution, which range from 2.0 to infinity.
    -radar_power:        Default 20 with var=1, only valid when noise_type is 'awgn+radar' and 'awgn+radar+denoise'.
    -radar_prob:         Default 0.05, only valid when noise_type is 'awgn+radar' and 'awgn+radar+denoise'.
    -denoise_thd:        Default 10, could change accordingly. only valid for 'awgn+radar+denoise'
    -fix_var             Default False. Using this argument with some float number of SNR in dB
(5) -snr_range:          Default -1.5 dB to 2 dB
(6) -snr_points:         Default 8
(7) -codec_type:         Default uses M=2, (7,5), feedback = 7. 'lte' uses LTE Turbo Codec. Codec has to match network model
(8) -num_cpu:            Number of CPU, by default use 5.
(*) --help:              Show Help Page

Example:
(1) To evaluate BER/BLER of AWGN Channel of Turbo Decoder, from -1.5 dB to 2 dB, 8 SNR points, with default setting.
python evaluate_commpy.py 

(2) To evaluate BER/BLER of AWGN Channel of Turbo Decoder with more blocks (Higher accuracy, avoid 0 in BER curve)
python evaluate_commpy.py -num_block 1000

(3) To evaluate BER/BLER of t-distribution Channel with v=3.0
python evaluate_commpy.py -num_block 1000 -noise_type t-dist -v 3.0 

(4) See Manual for evaluate_rnn.py
python evaluate_commpy.py --help


