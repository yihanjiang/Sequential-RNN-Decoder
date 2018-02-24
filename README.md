# Sequential-RNN-Decoder
Updated: 02/22/2018

This repository contains source code necessary to reproduce the results presented in the following paper:
Communication Algorithms via Deep Learning(https://openreview.net/pdf?id=ryazCMbR-) by Hyeji Kim, Yihan Jiang, Ranvir B. Rana, Sreeram Kannan, Sewoong Oh, Pramod Viswanath, accepted to ICLR 2018 as poster.

## Dependency
- Python (2.7.10+)
- numpy (1.14.1)
- Keras (2.0)
- scikit-commpy (0.3.0) For Commpy, we use a modified version of the original commpy, which is in the folder with name commpy. Commpy will be depreciated for future versions.
- h5py (2.7.0)
- tensorflow (1.5)

## Benchmarks
We have benchmarks for evaluating BER/BLER for convolutional code, turbo code. 
The curves from paper are from MATLAB simulation, the python curve is for reference. We find the python and MATLAB implementation has same performance.
When running large number of blocks (>1000), you might need to use multiprocess to speed up simulation, change -num_cpu to the number you like.

To evaluate BER/BLER for convolutional code, by default the codec is rate 1/2 (7,5) convolutioanl code with feedback = 7. (5 means f(x) = 1 + x^2, 7 means f(x)  = 1 + x + x^2)

    $ python conv_codes_benchmark.py -num_block 100 -block_len 100 -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8 -num_cpu 1

To evaluate BER/BLER for turbo code, by default the codec is rate 1/2 (7,5) convolutioanl code with feedback = 7. 

    $ python turbo_codes_benchmark.py -num_block 100 -block_len 100 -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8 -num_cpu 1

You can change to LTE turbo codec by

    $ python turbo_codes_benchmark.py -enc1 11 -enc2 13 -M 3 -feedback 11 -num_block 100 -block_len 100 -snr_test_start -1.5 -snr_test_end 2.0 -snr_points 8 -num_cpu 1
    
By default the number of decoding iteration is 6, you can change via change argument '-num_dec_iteration'


## RNN for Convolutioanl Code 
- use conv_decoder.py to train Viterbi-like RNNs. 
 
- use bcjr_rnn_train.py to train BCJR-like RNNs. 

## Neural Turbo Decoder
Note: Currently debugging non-AWGN decoders. Recommend to run on GPU. 

- To evaluate Neural Turbo Decoder run default setting by:
python turbo_neural_decoder_eval.py -h, to sepecify the parameters for testing. 
    
- To train Neural Turbo Decoder:
python turbo_neural_decoder_train.py -h, to specify the parameters for training

## Interpreting the RNN
Under construction.


<!--## Organization of codes-->
<!--(1) bcjr_util.py and utils.py:  Plan to merge. Utility Helpful Functions. \-->
<!--(2) turbo_RNN.py: Stacked Turbo RNN decoder. Plan to add TurboRNN Layer for further usage.\-->
<!--(3) model_zoo: trained models. \-->
<!--(4) commpy: Python Channel Codec.\-->
<!--(5) interface: usage for customized channel/decoder, etc.-->

## Questions?
Please email Yihan Jiang (yij021@uw.edu) for any questions.
