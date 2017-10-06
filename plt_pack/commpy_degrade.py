__author__ = 'yihanjiang'
import matplotlib.pylab as plt


def plt_awgn():

    SNRS      = [-1.00000012, -0.71346486 ,-0.41715208 ,-0.11037455 , 0.20763701 , 0.53773534, 0.88087624 , 1.2381326,   1.61071563,  2.00000024]

    BER_AWGN  = [  2.11690000e-03 ,  3.31300000e-04  , 7.24000000e-05,   2.80000000e-05,
                1.31000000e-05  , 6.50000000e-06  , 3.40000000e-06  , 1.60000000e-06 ,8.00000000e-07  , 6.00000000e-07]
    BER_radar = [ 0.208701  , 0.2027585 , 0.196706  , 0.191784 ,  0.1863105 , 0.1811835,
                    0.1767545,  0.172455 ,  0.1688995 , 0.1930065]

    BER_t5    = [  3.94050000e-03  , 1.64800000e-03 ,  7.77000000e-04 ,  3.74000000e-04,
                2.50000000e-04  , 1.03000000e-04 ,  4.45000000e-05 ,  2.60000000e-05, 1.70000000e-05 ,  8.00000000e-06]

    BER_t3   = [ 0.015285 ,  0.013918  , 0.013381 ,  0.01302  ,  0.0124915 , 0.0121825,
                0.0118775 , 0.012021 ,  0.0118625  ,0.011797 ]


    plt.figure(1)
    plt.title('Compare AWGN and ATN Turbo Performance')
    plt.xlabel('Channel SNR in dB')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid()
    p1, = plt.plot(SNRS, BER_AWGN, 'r', label = 'Turbo Decoding AWGN')
    #p2, = plt.plot(SNRS, BER_radar, 'g', label = 'Turbo Decoding Bursty Radar Noise')
    p3, = plt.plot(SNRS, BER_t5, 'b', label = 'Turbo Decoding ATN v = 5')
    p4, = plt.plot(SNRS, BER_t3, 'k', label = 'Turbo Decoding ATN v = 3')

    plt.legend(handles = [p1, p3, p4])
    plt.show()

    plt.figure(2)
    plt.title('Compare AWGN and Radar Noise Turbo Performance')
    plt.xlabel('Channel SNR in dB')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid()
    p1, = plt.plot(SNRS, BER_AWGN, 'r', label = 'Turbo Decoding AWGN')
    p2, = plt.plot(SNRS, BER_radar, 'g', label = 'Turbo Decoding Bursty Radar Noise')
    #p3, = plt.plot(SNRS, BER_t5, 'b', label = 'Turbo Decoding ATN v = 5')
    #p4, = plt.plot(SNRS, BER_t3, 'k', label = 'Turbo Decoding ATN v = 3')

    plt.legend(handles = [p1, p2])
    plt.show()


if __name__ == '__main__':
    plt_awgn()
