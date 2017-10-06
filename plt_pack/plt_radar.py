__author__ = 'yihanjiang'
import matplotlib.pylab as plt

def len1000_radar_model1():
    '''
    [testing] This SNR runnig time is 1210.01156211
[Result]SNR:  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
[Result]BER [0.068018999999999996, 0.010931, 0.00037100000000000002, 0.0001, 6.9999999999999999e-06, 3.9999999999999998e-06, 0.0, 0.0]
[Result]BLER [0.95799999999999996, 0.48999999999999999, 0.064999999999999947, 0.015000000000000013, 0.0030000000000000027, 0.0020000000000000018, 0.0, 0.0]
[Result]Total Running time: 10270.062916
    '''
    # Radar power = 20
    SNRS        = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    ber_denoise = [0.068018999999999996, 0.010931, 0.00037100000000000002, 0.0001, 6.9999999999999999e-06, 3.9999999999999998e-06, 0.0, 0.0]
    ber_commpy  = [0.028234, 0.002122, 6.8e-05, 2.01e-05, 7.1e-06, 2.9e-06, 9e-07, 3e-07]
    ber_radar   = [0.28123399999999998, 0.27918199999999999, 0.27749699999999999, 0.27626099999999998, 0.27601900000000001, 0.27487899999999998, 0.27381, 0.29230800000000001]
    rnn_radar_no_trained = [0.222, 0.1971, 0.18629999999999999, 0.16270000000000001, 0.1358, 0.13039999999999999, 0.11700000000000001, 0.1074]
    rnn_radar_trained    = [0.12909999999999999, 0.0332, 0.0079000000000000008, 0.00040000000000000002, 0.0, 0.00020000000000000001, 0.0, 0.0]



    plt.figure(1)
    plt.title('Block len = 1000 BER on variable length and iterations\n Comparing to Commpy Result')

    p10,  = plt.plot(SNRS, ber_denoise, 'y', label = 'Turbo Denoised Decoding')
    p11, = plt.plot(SNRS, ber_commpy, 'g',  label = 'Turbo AWGN Noise')
    p12, = plt.plot(SNRS, ber_radar, 'r', label = 'Turbo Radar Noise')
    p13, = plt.plot(SNRS, rnn_radar_no_trained, 'c-*', label = 'RNN not aware of Radar Noise')
    plt.legend(handles = [p10, p11,p12,   p13])

    plt.figure(2)
    plt.title('Block len = 1000 BER on variable length and iterations\n Comparing to Commpy Result')
    plt.yscale('log')

    p10,  = plt.plot(SNRS, ber_denoise, 'y', label = 'Turbo Denoised Decoding')
    p11, = plt.plot(SNRS, ber_commpy, 'g',  label = 'Turbo AWGN Noise')
    p12, = plt.plot(SNRS, ber_radar, 'r', label = 'Turbo Radar Noise')

    p20, = plt.plot(SNRS, rnn_radar_trained, 'c--', label = 'LSTM 6 iter')
    # p21, = plt.plot(SNRS, rnn_ber_bl1000_iter8, 'k--', label = 'LSTM 8 iter')
    # p22, = plt.plot(SNRS, rnn_ber_bl1000_iter4, 'b--', label = 'LSTM 4 iter')

    plt.legend(handles = [p10, p11,p12,   p20])
    plt.show()


def len1000_radar_model2():
    SNRS        =            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    commpy_ber_radar_20    = [0.48864999999999997, 0.50112000000000001, 0.50112000000000001, 0.50112000000000001, 0.50112000000000001, 0.50112000000000001, 0.50112000000000001, 0.50112000000000001]
    rnn_ber_trained_awgn_20= [0.20349999999999999, 0.1943, 0.16669999999999999, 0.14879999999999999, 0.1454, 0.12889999999999999, 0.108, 0.1002]
    rnn_ber_trained_m1_20  = [0.15790000000000001, 0.1197, 0.0516, 0.028400000000000002, 0.0109, 0.0057999999999999996, 0.0056999999999999999, 0.0048999999999999998]

    commpy_ber_radar_10    = [0.1888, 0.17480000000000001, 0.1961, 0.18759999999999999, 0.17549999999999999, 0.20349999999999999, 0.22750000000000001, 0.22900000000000001]
    rnn_ber_trained_awgn_10= [0.2112, 0.18814, 0.16841, 0.14932000000000001, 0.13252, 0.12039, 0.11054, 0.10215]
    rnn_ber_trained_m1_10  = [0.14330000000000001, 0.112, 0.073800000000000004, 0.032300000000000002, 0.019300000000000001, 0.0058999999999999999, 0.0086, 0.0053]

    plt.figure(1)
    plt.title('Model 2 Block len = 1000 Radar Power 20')
    plt.yscale('log')
    p10,  = plt.plot(SNRS, commpy_ber_radar_20, 'y', label = 'Turbo Decoder')
    p11, = plt.plot(SNRS, rnn_ber_trained_awgn_20, 'g',  label = 'RNN Not trained')
    p12, = plt.plot(SNRS, rnn_ber_trained_m1_20, 'r', label = 'RNN trained with model 1')
    plt.legend(handles = [p10, p11,p12])

    plt.figure(2)
    plt.title('Model 2 Block len = 1000 Radar Power 10')
    plt.yscale('log')
    p10,  = plt.plot(SNRS, commpy_ber_radar_10, 'y', label = 'Turbo Decoder')
    p11, = plt.plot(SNRS, rnn_ber_trained_awgn_10, 'g',  label = 'RNN Not trained')
    p12, = plt.plot(SNRS, rnn_ber_trained_m1_10, 'r', label = 'RNN trained with model 1')
    plt.legend(handles = [p10, p11,p12])


    plt.show()




def len100_radar_model1():
    pass

def len100_radar_model2():
    pass

def len1000_radar_model_onAWGN():

    SNRS        =            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    RNN_radar_trained = [0.036401999999999997, 0.0034880000000000002, 0.00012549999999999999, 2.65e-05,
                         6.0000000000000002e-06, 2.8e-06, 0.0, 9.9999999999999995e-07]
    rnn_ber_bl1000_iter6    = [0.036448999999999998, 0.00348, 0.000116, 2.49e-05, 8.74e-06, 3.48e-06, 1.46e-06, 3.4e-07]
    ber_commpy  = [0.028234, 0.002122, 6.8e-05, 2.01e-05, 7.1e-06, 2.9e-06, 9e-07, 3e-07]

    RNN_radar_trained_bler = [0.00145]


    plt.figure(1)
    plt.title('BER on AWGN Block len = 1000 ')
    plt.yscale('log')
    p10,  = plt.plot(SNRS, RNN_radar_trained, 'y', label = 'RNN Radar Trained')
    p11, = plt.plot(SNRS, rnn_ber_bl1000_iter6, 'g',  label = 'RNN AWGN trained')
    p12, = plt.plot(SNRS, ber_commpy, 'r', label = 'Commpy')
    plt.legend(handles = [p10, p11,p12])
    plt.show()

if __name__ == '__main__':
    len1000_radar_model2()
    #len1000_radar_model_onAWGN()