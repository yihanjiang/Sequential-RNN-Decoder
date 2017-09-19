# TurboRNNDecoder
RNN Turbo Decoder
Author: Yihan Jiang, UW & Hyeji Kim, UIUC.

We have a few scripts for user:

(1) evaluate_rnn.py is a command line tool for evaluating Turbo RNN Decoder's BER curve.

(2) evaluate_rnn.py is a command line tool for evaluating Turbo RNN Decoder's BER curve.

(2) train_turbo_decoder_end2end.py is a command line tool for training Turbo Decoder with different noise.

(3) train_bcjr_rnn.py is a command line tool for training BCJR-like RNN

(4) TBD: interpretibility.py

(*) TBD: Support User Defined Codec

(*) TBD: User Defined Channel, or let user override channel easily.
 

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


Organization of codes:
(1) bcjr_util.py and utils.py:  Plan to merge. Utility Helpful Functions. 

(2) turbo_RNN.py: Stacked Turbo RNN decoder. Plan to add TurboRNN Layer for further usage.

(3) commpy_turbo_infer.py: Use Commpy to get BER for Normal Turbo Decoding.

(4) interpret.py: Interpret module, try to understand what we have learned.

(5) train_bcjr_rnn.py: Init the Turbo Decoder with some good init. 