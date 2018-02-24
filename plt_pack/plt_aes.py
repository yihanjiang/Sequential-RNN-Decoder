__author__ = 'yihanjiang'

import matplotlib.pylab as plt


snrs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

ber_uncoded = [0.104094, 0.078776, 0.057114, 0.037718, 0.022926, 0.01239, 0.005908, 0.002374, 0.000774, 0.000174]
bler_uncoded= [0.9971, 0.9812, 0.9473, 0.8566, 0.681, 0.4636, 0.2539, 0.11070000000000002, 0.037900000000000045, 0.008700000000000041]

ber_ae      =  [0.104573, 0.0720995, 0.04538975, 0.02554425, 0.012568, 0.00512625, 0.00175375, 0.0004645, 9.95e-05, 1.325e-05]
bler_ae      =  [0.999575, 0.996, 0.97065, 0.868925, 0.645275, 0.362975, 0.146775, 0.042825, 0.009575, 0.0013]

# CUDA_VISIBLE_DEVICES=0 python last2_var_block.py -num_block 10000 -batch_size 10 -enc_direction bd  -block_len 100 -block_noise_prob 0.8
ber_ae = [0.109343, 0.076096, 0.047661, 0.02694, 0.01325, 0.005495, 0.001835, 0.000554, 0.000114, 1.2e-05]
bler_ae      = [0.9994, 0.9971, 0.9722, 0.8724, 0.6507, 0.3682, 0.1491, 0.0503, 0.0111, 0.0012]

ber_conv_75 = [0.174721000000000	,0.106475000000000	,0.0545840000000000	,0.0203690000000000	,0.00608500000000000	,0.00153200000000000	,0.000325000000000000	,5.70000000000000e-05	,9.00000000000000e-06,	1.00000000000000e-06]
bler_conv_75 = [0.9949, 0.956, 0.8047, 0.5129, 0.248, 0.11219999999999997, 0.06159999999999999, 0.032200000000000006, 0.01749999999999996, 0.011600000000000055]

#python conv_codes.py -num_cpu 40 -enc1 7 -enc2 5 -feedback 0 -code_rate 2 -block_len 100 -num_block 1000000
ber_conv_75 =  [0.15549769, 0.0903642, 0.04148792, 0.01429561, 0.00357761, 0.00063561, 7.874e-05, 6.74e-06, 3.7e-07, 2e-08]
bler_conv_75 =  [0.993433, 0.944145, 0.75444, 0.421173, 0.14996500000000001, 0.03427100000000005, 0.005118999999999985, 0.0005129999999999857, 3.2999999999949736e-05, 1.0000000000287557e-06]

# python last2_loss_finetune.py -enc_direction bd -block_len 100 -bsc_prob 0.0 -enc_num_unit 100 -enc_num_layer 2 -dec_num_layer 2 -num_block 40000 -batch_size 100 -init_nw_weight ./tmp/603003.h5 -train_channel_low -1.51 -train_channel_high -1.5
ber_ae = [0.104373, 0.0719865, 0.045222, 0.02546475, 0.01242225, 0.0051585, 0.0017165, 0.00045725, 8.55e-05, 9.75e-06]
bler_ae = [0.9997, 0.996525, 0.969225, 0.8703, 0.64205, 0.361525, 0.143425, 0.0424, 0.008275, 0.00095]

#CUDA_VISIBLE_DEVICES=3 python last2_var_block.py -block_noise_prob 0.8 -enc_direction bd -block_len 100 -bsc_prob 0.0 -enc_num_unit 100 -enc_num_layer 2 -dec_num_layer 4 -num_block 20000 -batch_size 100  -init_nw_weight ./tmp/581422.h5
#./tmp/057234.h5
ber_ae = [0.1090115, 0.07459, 0.0464545, 0.025404, 0.0121455, 0.004865, 0.00155, 0.000416, 8.55e-05, 9e-06]
bler_ae = [0.9994, 0.9953, 0.96645, 0.85025, 0.6147, 0.32805, 0.12655, 0.037, 0.00805, 0.0009]


#python conv_decoder.py -batch_size 200 -learning_rate 0.0001 -train_channel_low -1.5 -train_channel_high -1.49 -num_block 500000 -block_len 100 -Dec_weight ./tmp/conv_dec320446.h5
ber_NN_conv_decoder = [0.1384904, 0.0919964, 0.0530316, 0.0256448, 0.0098256, 0.0030374, 0.0007654, 0.0001846, 4.56e-05, 6.2e-06]
bler_NN_conv_decoder= [0.99906, 0.9886, 0.93064, 0.74192, 0.42846, 0.16772, 0.0465, 0.01086, 0.0026, 0.00038]

plt.figure(1)
plt.subplot(121)
plt.yscale('log')
p1, = plt.plot(snrs,ber_conv_75, label = 'conv_75' )
p2, = plt.plot(snrs, ber_ae, label = 'ae')
p3, = plt.plot(snrs, ber_uncoded, label = 'uncoded')
p4, = plt.plot(snrs, ber_NN_conv_decoder, label = 'nn_conv_75')
plt.xlabel('ber')
plt.legend(handles = [p1, p2,p3, p4])

plt.subplot(122)
plt.yscale('log')
p1, = plt.plot(snrs,bler_conv_75, label = 'conv_75' )
p2, = plt.plot(snrs, bler_ae, label = 'ae')
p3, = plt.plot(snrs, bler_uncoded, label = 'uncoded')
p4, = plt.plot(snrs, bler_NN_conv_decoder, label = 'nn_conv_75')
plt.legend(handles = [p1, p2,p3, p4])
plt.xlabel('bler')
plt.show()