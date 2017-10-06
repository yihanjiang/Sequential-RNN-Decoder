__author__ = 'yihanjiang'
import matplotlib.pylab as plt

def t_dist():
    SNRS =   [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    commpy_ber_t3 = [0.022020000000000001, 0.016760000000000001, 0.014659999999999999, 0.013679999999999999, 0.013100000000000001, 0.01286, 0.01259, 0.012449999999999999]
    commpy_bler_t3= [0.68500000000000005, 0.505, 0.38, 0.26500000000000001, 0.18500000000000005, 0.14000000000000001, 0.094999999999999973, 0.064999999999999947]

    commpy_ber_t5 = [0.017777999999999999, 0.0041609999999999998, 0.00097499999999999996, 0.00026200000000000003, 9.8999999999999994e-05, 2.4000000000000001e-05, 1.5999999999999999e-05, 9.0000000000000002e-06]
    commpy_bler_t5= [0.71899999999999997, 0.34099999999999997, 0.13400000000000001, 0.050000000000000044, 0.028000000000000025, 0.01100000000000001, 0.0060000000000000053, 0.0030000000000000027]

    rnn_awgn_ber_t3 = [0.022172999999999998, 0.0044840000000000001, 0.00086850000000000002, 0.00016200000000000001, 5.8499999999999999e-05, 2.8e-05, 1.3499999999999999e-05, 7.5000000000000002e-06]
    rnn_awgn_bler_t3 =[0.84250000000000003, 0.45300000000000001, 0.187, 0.064500000000000002, 0.028000000000000001, 0.017000000000000001, 0.0080000000000000002, 0.0040000000000000001]

    rnn_awgn_ber_t5 = [0.0232505, 0.0046845000000000003, 0.00082950000000000005, 0.00019900000000000001, 6.3999999999999997e-05, 3.3500000000000001e-05, 2.3499999999999999e-05, 1.7e-05]
    rnn_awgn_bler_t5 =[0.84399999999999997, 0.47249999999999998, 0.19, 0.071999999999999995, 0.032500000000000001, 0.019, 0.0115, 0.0089999999999999993]

    rnn_train_ber_t3 =[0.0219044999999999999, 0.0044710000000000002, 0.00097900000000000005, 0.00021800000000000001, 4.6999999999999997e-05, 1.4e-05, 1.1e-05, 6.9999999999999999e-06]
    rnn_train_bler_t3 =[0.88200000000000001, 0.48499999999999999, 0.185, 0.058999999999999997, 0.019, 0.0060000000000000001, 0.0040000000000000001, 0.0040000000000000001]

    rnn_train_ber_t5 =[0.014807000000000001, 0.0025969999999999999, 0.00035799999999999997, 0.00010900000000000001, 4.3000000000000002e-05, 1.1e-05, 7.9999999999999996e-06, 7.9999999999999996e-06]
    rnn_train_bler_t5 =[0.71399999999999997, 0.29299999999999998, 0.10100000000000001, 0.041000000000000002, 0.021000000000000001, 0.0060000000000000001, 0.0040000000000000001, 0.0020000000000000001]

    plt.figure(1)

    plt.subplot(121)
    plt.title('T-distribution on v=3 BER')
    plt.yscale('log')
    p1, = plt.plot(SNRS, commpy_ber_t3, label = 'Commpy Decoding')
    p2, = plt.plot(SNRS, rnn_awgn_ber_t3, label = 'RNN AWGN Decoding')
    p3, = plt.plot(SNRS, rnn_train_ber_t3, label = 'RNN t-distribution Decoding')
    plt.legend(handles = [p1, p2, p3])
    plt.grid()

    plt.subplot(122)
    plt.title('T-distribution on v=5 BER')
    plt.yscale('log')
    p1, = plt.plot(SNRS, commpy_ber_t5, label = 'Commpy Decoding')
    p2, = plt.plot(SNRS, rnn_awgn_ber_t5, label = 'RNN AWGN Decoding')
    p3, = plt.plot(SNRS, rnn_train_ber_t5, label = 'RNN t-distribution Decoding')
    plt.legend(handles = [p1, p2, p3])
    plt.grid()

    plt.figure(2)

    plt.subplot(121)
    plt.title('T-distribution on v=3 BLER')
    plt.yscale('log')
    p1, = plt.plot(SNRS, commpy_bler_t3, label = 'Commpy Decoding')
    p2, = plt.plot(SNRS, rnn_awgn_bler_t3, label = 'RNN AWGN Decoding')
    p3, = plt.plot(SNRS, rnn_train_bler_t3, label = 'RNN t-distribution Decoding')
    plt.legend(handles = [p1, p2, p3])
    plt.grid()

    plt.subplot(122)
    plt.title('T-distribution on v=5 BLER')
    plt.yscale('log')
    p1, = plt.plot(SNRS, commpy_bler_t5, label = 'Commpy Decoding')
    p2, = plt.plot(SNRS, rnn_awgn_bler_t5, label = 'RNN AWGN Decoding')
    p3, = plt.plot(SNRS, rnn_train_bler_t5, label = 'RNN t-distribution Decoding')
    plt.legend(handles = [p1, p2, p3])
    plt.grid()

    plt.show()





if __name__ == '__main__':
    t_dist()