## Sequential-RNN-Decoder
**RNN Turbo Decoder**
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
(1) To evaluate Neural Turbo Decoder run default setting by:

python turbo_neural_decoder_eval.py 

To train Neural Turbo Decoder


To train Neural Turbo Decoder 


We have a few scripts for user: 

(1) **evaluate_rnn.py** is a command line tool for evaluating Turbo RNN Decoder's BER curve.\
(2) **evaluate_commpy.py** is a command line tool for evaluating Turbo Commpy Decoder's BER curve.\
(3) **train_turbo_decoder_end2end.py** is a command line tool for training Turbo Decoder with different noise.\
(4) **train_bcjr_rnn.py** is a command line tool for training BCJR-like RNN.\
(5) **interpret.py** shows how to duplicate interpretable graph in paper.\
(7) TBD: Support User Defined Codec.\
(8) TBD: User Defined Channel, or let user override channel easily.
 


Organization of codes:\
(1) bcjr_util.py and utils.py:  Plan to merge. Utility Helpful Functions. \
(2) turbo_RNN.py: Stacked Turbo RNN decoder. Plan to add TurboRNN Layer for further usage.\
(3) model_zoo: trained models. \
(4) commpy: Python Channel Codec.\
(5) interface: usage for customized channel/decoder, etc.
