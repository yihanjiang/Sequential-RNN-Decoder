## Sequential-RNN-Decoder
Updated: 02/22/2018

This repository contains source code necessary to reproduce the results presented in the following paper:
Communication Algorithms via Deep Learning(https://openreview.net/pdf?id=ryazCMbR-) by Hyeji Kim, Yihan Jiang, Ranvir B. Rana, Sreeram Kannan, Sewoong Oh, Pramod Viswanath, accepted to ICLR 2018 as poster.

# Dependency
(0) Python (2.7.10+)\
(1) numpy (1.14.1+)\
(2) Keras (2.0+)\
(3) scikit-commpy (0.3.0) For Commpy, we use a modified version of the original commpy, which is in the folder with name commpy. Commpy will be depreciated for future versions.\
(4) h5py (2.7.0+)\
(5) tensorflow (1.2+)\
Use pip to install above packages.

# RNN for Convolutioanl Code 
(1)




(2) Train BCJR-like

# Neural Turbo Decoder
Currently debugging non-AWGN decoders. AWGN decoders works well.

(1) To evaluate Neural Turbo Decoder run default setting by:
python turbo_neural_decoder_eval.py -h, to sepecify the parameters for testing. The following command will test Turbo Neural Decoder with block length 100 between -1.5dB to 2dB for each 0.5 dB (8 points) with 100 blocks. The default model is for block length 100 AWGN neural decoder. If test on block length 1000, please use

    $ python turbo_neural_decoder_eval.py -num_block 100 -block_len 100 \
    -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8 -model_path ./models/turbo_models/awgn_bl100_1014.h5

(2) To train Neural Turbo Decoder:
python turbo_neural_decoder_train.py -h, to specify the parameters for training

# Interpreting the RNN
Under construction.


# Benchmarks
We have benchmarks for evaluating BER/BLER for convolutional code, turbo code. 
LDPC/Polar code under construction.

We have a few scripts for user: 

(1) **evaluate_rnn.py** is a command line tool for evaluating Turbo RNN Decoder's BER curve.\
(2) **evaluate_commpy.py** is a command line tool for evaluating Turbo Commpy Decoder's BER curve.\
(3) **train_turbo_decoder_end2end.py** is a command line tool for training Turbo Decoder with different noise.\
(4) **train_bcjr_rnn.py** is a command line tool for training BCJR-like RNN.\
(5) **interpret.py** shows how to duplicate interpretable graph in paper.\
(7) TBD: Support User Defined Codec.\
(8) TBD: User Defined Channel, or let user override channel easily.
 

# Organization of codes
(1) bcjr_util.py and utils.py:  Plan to merge. Utility Helpful Functions. \
(2) turbo_RNN.py: Stacked Turbo RNN decoder. Plan to add TurboRNN Layer for further usage.\
(3) model_zoo: trained models. \
(4) commpy: Python Channel Codec.\
(5) interface: usage for customized channel/decoder, etc.

# Questions?
Please email Yihan Jiang (yij021@uw.edu) for any questions.
