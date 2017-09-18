__author__ = 'yihanjiang'

'''
Interpretibility Module.Not a command line tool!
'''

import math
import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv
import commpy.channelcoding.turbo as turbo
import matplotlib.pylab as plt

import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional

from utils import corrupt_signal, build_rnn_data_feed

class Interpret(object):
    def __init__(self,network_saved_path,
                 num_block = 100,
                 block_len = 100,num_hidden_unit = 200 ):

        self.block_len = block_len
        self.num_block = num_block
        self.network_saved_path = network_saved_path
        self.model = self._load_model()

    def delete_model(self):
        del self.model

    def _load_model(self,  num_hidden_unit = 200):
        '''
        Only works for interpretibility, you have to know: num_hidden_unit.
        :param block_len:
        :param network_saved_path:
        :param learning_rate:
        :param num_hidden_unit:
        :return:
        '''
        if self.network_saved_path == 'default':
            network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
        else:
            network_saved_path = self.network_saved_path

        rnn_type    = 'lstm'    #'gru', 'lstm'
        print '[BCJR RNN Interpret] using model type', rnn_type
        print '[BCJR RNN Interpret] using model path', network_saved_path

        batch_size    = 32

        print '[RNN Model] Block length', self.block_len
        print '[RNN Model] Evaluate Batch size', batch_size

        def errors(y_true, y_pred):
            myOtherTensor = K.not_equal(y_true, K.round(y_pred))
            return K.mean(tf.cast(myOtherTensor, tf.float32))

        ####################################################
        # Define Model
        ####################################################
        if rnn_type == 'lstm':
            f1 = Bidirectional(LSTM(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(LSTM(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')
        else:
            f1 = Bidirectional(GRU(name='bidirectional_1', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f2 = BatchNormalization(name='batch_normalization_1')
            f3 = Bidirectional(GRU(name='bidirectional_2', units=num_hidden_unit, activation='tanh', return_sequences=True, dropout=1.0))
            f4 = BatchNormalization(name='batch_normalization_2')

        f5 = TimeDistributed(Dense(1, activation='linear'),name='time_distributed_1')

        inputs = Input(shape = (self.block_len,3))
        predictions = f5(f4(f3(f2(f1(inputs)))))

        model = Model(inputs=inputs, outputs=predictions)
        optimizer= keras.optimizers.adam(lr=0.001, clipnorm=1.0)               # not useful
        model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])
        model.load_weights(network_saved_path, by_name=True)

        return model

    def likelihood(self, bit_pos, sigma, radar_noise_power=20.0, is_compute_no_bursty = False, is_compute_map = False):
        '''
        compute the likelihood along the positions.
        :param bit_pos:
        :param sigma:
        :param radar_noise_power:
        :param is_compute_no_bursty:
        :param is_compute_map:
        :return:
        '''

        model = self.model

        # Define the code, too zy, reuse the code
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])
        feedback = 7

        trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        interleaver = RandInterlv.RandInterlv(self.block_len, 0)

        noiser = ['awgn', sigma]
        codec  = [trellis1, trellis2, interleaver]
        X_feed_test, X_message_test = build_rnn_data_feed(self.num_block, self.block_len, noiser, codec, is_all_zero=True)

        if is_compute_no_bursty:
            ##############################################################
            # Non Bursty Noise Case
            # RNN Decoder
            ##############################################################
            pd                = model.predict(X_feed_test[:, :, :3],batch_size = 100)
            rnn_ll_non_bursty = np.mean(pd, axis = 0).T.tolist()[0]

            if is_compute_map:
                # BCJR/MAP Decoder
                map_likelihood_list = []
                for block_idx in range(self.num_block):
                    message_bits =  X_message_test[block_idx, :, :]
                    encoded      =  X_feed_test[block_idx, :, :2]
                    sys_r        =  encoded[:, 0]
                    par1_r       =  encoded[:, 1]

                    # CommPy Map
                    L_int = np.zeros(len(sys_r))

                    [L_ext_1_noburst, decoded_bits] = turbo.map_decode(sys_r, par1_r,
                                                         trellis1, sigma**2, L_int, 'decode')

                    map_likelihood_list.append(L_ext_1_noburst)

                map_ll_non_bursty = np.stack(np.array(map_likelihood_list), axis=0)
                map_ll_non_bursty = np.mean(map_ll_non_bursty, axis=0).T.tolist()


        # Bursty Noise Case
        radar_noise = np.zeros(X_feed_test.shape)
        # mind!
        radar_noise[:, bit_pos, :] = radar_noise_power
        X_feed_test += radar_noise

        # RNN Decoder
        pd     = model.predict(X_feed_test[:, :, :3],batch_size = 100)
        rnn_ll_bursty = np.mean(pd, axis = 0).T.tolist()[0]

        if is_compute_map:
            # BCJR/MAP Decoder
            map_likelihood_list = []
            for block_idx in range(self.num_block):
                message_bits =  X_message_test[block_idx, :, :]
                encoded      =  X_feed_test[block_idx, :, :2]
                sys_r        =  encoded[:, 0]
                par1_r       =  encoded[:, 1]

                # CommPy Map
                L_int = np.zeros(len(sys_r))

                [L_ext_1_noburst, decoded_bits] = turbo.map_decode(sys_r, par1_r,
                                                     trellis1, sigma**2,  L_int, 'decode')

                map_likelihood_list.append(L_ext_1_noburst)

            map_ll_bursty = np.stack(np.array(map_likelihood_list), axis=0)
            map_ll_bursty = np.mean(map_ll_bursty, axis=0).T.tolist()

        if is_compute_no_bursty:
            if is_compute_map:
                return map_ll_non_bursty, rnn_ll_non_bursty, map_ll_bursty, rnn_ll_bursty
            else:
                return rnn_ll_non_bursty, rnn_ll_bursty
        else:
            if is_compute_map:
                return map_ll_bursty, rnn_ll_bursty
            else:
                return rnn_ll_bursty

    def ber(self, bit_pos, noise_power):
        '''
        Compute BER along block in different positions.
        :param bit_pos:
        :param noise_power:
        :return:
        '''
        model = self.model

        # Define the code, too zy, reuse the code
        M = np.array([2]) # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])
        feedback = 7

        trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
        interleaver = RandInterlv.RandInterlv(self.block_len, 0)

        noiser = ['awgn', sigma]
        codec  = [trellis1, trellis2, interleaver]
        X_feed_test, X_message_test = build_rnn_data_feed(self.num_block, self.block_len, noiser, codec, is_all_zero=True)


def likelihood_snr_range():
    print '[Interpret] Likelihood output of BCJR/RNN'
    ###############################################
    # Input Parameters
    ###############################################
    #network_saved_path = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
    network_saved_path = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_1.h5'
    interpret = Interpret(network_saved_path=network_saved_path, block_len=100, num_block=1000)

    radar_bit_pos = 50

    map_ll_non_bursty1, rnn_ll_non_bursty1, map_ll_bursty1, rnn_ll_bursty1 = interpret.likelihood(bit_pos=radar_bit_pos, sigma=0.5,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True)

    map_ll_non_bursty2, rnn_ll_non_bursty2, map_ll_bursty2, rnn_ll_bursty2 = interpret.likelihood(bit_pos=radar_bit_pos,sigma = 1.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True)

    map_ll_non_bursty3, rnn_ll_non_bursty3, map_ll_bursty3, rnn_ll_bursty3 = interpret.likelihood(bit_pos=radar_bit_pos,sigma = 2.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True)

    map_ll_non_bursty4, rnn_ll_non_bursty4, map_ll_bursty4, rnn_ll_bursty4 = interpret.likelihood(bit_pos=radar_bit_pos,sigma = 5.0,
                                                                                              radar_noise_power = 10, is_compute_map=True,
                                                                                              is_compute_no_bursty=True)

    plt.figure(1)
    plt.subplot(121)
    plt.title('likelihood compare on different snr, RNN, Only AWGN Noise')
    p1, = plt.plot(rnn_ll_non_bursty1, 'y', label ='RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_non_bursty2, 'g', label ='RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_non_bursty3, 'b', label ='RNN AWGNsigma 2.0')
    p4, = plt.plot(rnn_ll_non_bursty4, 'r', label ='RNN AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])

    plt.subplot(122)
    plt.title('likelihood compare on different snr, BCJR/MAP, Only AWGN Noise')
    p1, = plt.plot(map_ll_non_bursty1, 'y', label ='BCJR AWGN sigma 0.5' )
    p2, = plt.plot(map_ll_non_bursty2, 'g', label ='BCJR AWGN sigma 1.0')
    p3, = plt.plot(map_ll_non_bursty3, 'b', label ='BCJR AWGNsigma 2.0')
    p4, = plt.plot(map_ll_non_bursty4, 'r', label ='BCJR AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])
    plt.show()

    plt.figure(2)
    plt.subplot(121)
    plt.title('likelihood compare on different snr, RNN, Radar Noise')
    p1, = plt.plot(rnn_ll_bursty1, 'y', label ='RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_bursty2, 'g', label ='RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_bursty3, 'b', label ='RNN AWGNsigma 2.0')
    p4, = plt.plot(rnn_ll_bursty4, 'r', label ='RNN AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])

    plt.subplot(122)
    plt.title('likelihood compare on different snr, BCJR/MAP, Radar Noise')
    p1, = plt.plot(map_ll_bursty1, 'y', label ='BCJR AWGN sigma 0.5' )
    p2, = plt.plot(map_ll_bursty2, 'g', label ='BCJR AWGN sigma 1.0')
    p3, = plt.plot(map_ll_bursty3, 'b', label ='BCJR AWGNsigma 2.0')
    p4, = plt.plot(map_ll_bursty4, 'r', label ='BCJR AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p4])
    plt.show()


def likelihood_radartrained_vs_awgntrained():
    ###############################################
    # Input Parameters
    ###############################################
    network_saved_path_awgntrained = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_3.h5'
    #network_saved_path_radartrained = './model_zoo/radar_model_end2end/0911radar_end2end_ttbl_0.406623492103_snr_1.h5'
    network_saved_path_radartrained = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_4.h5'

    radar_bit_pos = 50
    #
    #
    # interpret_awgn  = Interpret(network_saved_path=network_saved_path_awgntrained, block_len=100, num_block=1000)
    # rnn_ll_non_bursty1, rnn_ll_bursty1 = interpret_awgn.likelihood(bit_pos=radar_bit_pos,sigma = 0.5,
    #                                                               radar_noise_power = 10, is_compute_map=False,
    #                                                               is_compute_no_bursty=True)
    # rnn_ll_non_bursty2, rnn_ll_bursty2 = interpret_awgn.likelihood(bit_pos=radar_bit_pos,sigma = 1.0,
    #                                                               radar_noise_power = 10, is_compute_map=False,
    #                                                               is_compute_no_bursty=True)
    # rnn_ll_non_bursty3, rnn_ll_bursty3 = interpret_awgn.likelihood(bit_pos=radar_bit_pos,sigma = 2.0,
    #                                                               radar_noise_power = 10, is_compute_map=False,
    #                                                               is_compute_no_bursty=True)
    # rnn_ll_non_bursty4, rnn_ll_bursty4 = interpret_awgn.likelihood(bit_pos=radar_bit_pos,sigma = 5.0,
    #                                                               radar_noise_power = 10, is_compute_map=False,
    #                                                               is_compute_no_bursty=True)
    #
    # print rnn_ll_non_bursty1
    # print rnn_ll_bursty1
    # print rnn_ll_non_bursty2
    # print rnn_ll_bursty2
    # print rnn_ll_non_bursty3
    # print rnn_ll_bursty3
    # print rnn_ll_non_bursty4
    # print rnn_ll_bursty4

    rnn_ll_non_bursty1 = [-2.718410015106201, -2.238192081451416, -2.2305047512054443, -2.020719289779663, -1.9832576513290405, -1.9248669147491455, -1.8895856142044067, -1.8433572053909302, -1.8303667306900024, -1.8162685632705688, -1.7807395458221436, -1.7705918550491333, -1.7679224014282227, -1.7744837999343872, -1.7788676023483276, -1.7759292125701904, -1.7764670848846436, -1.7821993827819824, -1.778914451599121, -1.7770512104034424, -1.7839255332946777, -1.7789411544799805, -1.7840632200241089, -1.7826499938964844, -1.7902342081069946, -1.7917927503585815, -1.7868351936340332, -1.8102456331253052, -1.7914340496063232, -1.7813812494277954, -1.7737486362457275, -1.774087905883789, -1.7707104682922363, -1.7799075841903687, -1.7681996822357178, -1.753780484199524, -1.749144434928894, -1.7702959775924683, -1.7731297016143799, -1.7716792821884155, -1.771443486213684, -1.76394784450531, -1.7753181457519531, -1.761722445487976, -1.7432230710983276, -1.7396639585494995, -1.7434353828430176, -1.7411425113677979, -1.7385185956954956, -1.75456964969635, -1.7759740352630615, -1.7602213621139526, -1.7536985874176025, -1.7610321044921875, -1.7392656803131104, -1.7623896598815918, -1.7458628416061401, -1.763697624206543, -1.7530802488327026, -1.75632905960083, -1.7474706172943115, -1.7401821613311768, -1.7447609901428223, -1.7457568645477295, -1.7573456764221191, -1.748127818107605, -1.7510724067687988, -1.766837239265442, -1.7543495893478394, -1.753349781036377, -1.7549958229064941, -1.758803367614746, -1.74042809009552, -1.751119613647461, -1.742134690284729, -1.7332102060317993, -1.7360527515411377, -1.7327044010162354, -1.7294340133666992, -1.7114402055740356, -1.7173259258270264, -1.7002689838409424, -1.6758065223693848, -1.6773335933685303, -1.6634304523468018, -1.6420700550079346, -1.6306116580963135, -1.6260191202163696, -1.6090171337127686, -1.579788088798523, -1.552893877029419, -1.5188782215118408, -1.4745221138000488, -1.4465941190719604, -1.3958970308303833, -1.340663194656372, -1.255033016204834, -1.189096450805664, -1.0162566900253296, -0.7628799676895142]
    rnn_ll_bursty1 = [-2.7267422676086426, -2.245661735534668, -2.232862710952759, -2.0210018157958984, -1.9828908443450928, -1.9235063791275024, -1.8878740072250366, -1.8412964344024658, -1.827757716178894, -1.8130334615707397, -1.776963233947754, -1.766445517539978, -1.7635434865951538, -1.7692699432373047, -1.773674726486206, -1.7699352502822876, -1.769695520401001, -1.7744039297103882, -1.7699456214904785, -1.7670236825942993, -1.773094892501831, -1.7669425010681152, -1.770297646522522, -1.7667902708053589, -1.7730265855789185, -1.7718479633331299, -1.7644450664520264, -1.785400629043579, -1.76276695728302, -1.7483100891113281, -1.7364296913146973, -1.7306724786758423, -1.7210429906845093, -1.7216473817825317, -1.7011219263076782, -1.6723198890686035, -1.6556894779205322, -1.6561245918273926, -1.6413222551345825, -1.6048628091812134, -1.5837913751602173, -1.5169247388839722, -1.5073779821395874, -1.3999484777450562, -1.3744993209838867, -1.227081298828125, -1.2941845655441284, -1.005530595779419, -1.2146416902542114, -1.1289817094802856, 4.897057056427002, 1.6852805614471436, -0.5479801893234253, -0.4787071645259857, -0.9292492866516113, -1.0357420444488525, -1.2444039583206177, -1.3346389532089233, -1.448879361152649, -1.510166883468628, -1.5716383457183838, -1.604288101196289, -1.6470826864242554, -1.6695971488952637, -1.7038627862930298, -1.7064770460128784, -1.7216901779174805, -1.7446675300598145, -1.738376498222351, -1.7414593696594238, -1.7461117506027222, -1.7522903680801392, -1.73549222946167, -1.7473965883255005, -1.739261507987976, -1.7309659719467163, -1.7342323064804077, -1.7312242984771729, -1.7282137870788574, -1.7103922367095947, -1.7163997888565063, -1.6994662284851074, -1.6750385761260986, -1.6766324043273926, -1.6627862453460693, -1.641433835029602, -1.6300060749053955, -1.6254328489303589, -1.6084344387054443, -1.5792174339294434, -1.5523232221603394, -1.5183234214782715, -1.473942518234253, -1.4460294246673584, -1.3953158855438232, -1.3400793075561523, -1.2544059753417969, -1.1884891986846924, -1.0156573057174683, -0.7629398703575134]
    rnn_ll_non_bursty2 = [-2.1625964641571045, -1.4361711740493774, -1.4237067699432373, -1.232788324356079, -1.1677991151809692, -1.120484471321106, -1.0750600099563599, -1.0610562562942505, -1.0318681001663208, -1.039977788925171, -1.03008234500885, -1.0186827182769775, -0.9952101111412048, -1.0107241868972778, -1.0031589269638062, -1.0089805126190186, -1.0069177150726318, -1.0052546262741089, -1.0122239589691162, -1.0316444635391235, -0.9772986173629761, -1.0030169486999512, -1.0186351537704468, -1.0055588483810425, -1.0145403146743774, -1.0515997409820557, -1.0495107173919678, -1.005632758140564, -1.0328961610794067, -1.0431209802627563, -1.063456654548645, -1.0217769145965576, -1.0406873226165771, -1.028833031654358, -1.0470192432403564, -1.020633339881897, -1.0286616086959839, -0.9768626093864441, -1.0110969543457031, -0.9955248236656189, -1.026292324066162, -0.9878073930740356, -1.0288978815078735, -0.9873716831207275, -1.0257059335708618, -1.0065441131591797, -1.0438473224639893, -1.002202033996582, -0.9919758439064026, -0.9814913272857666, -1.0037283897399902, -0.972831130027771, -0.9476854801177979, -0.9545490741729736, -1.013339638710022, -0.9968898296356201, -0.9872196316719055, -0.976678729057312, -0.9695214033126831, -0.967581033706665, -0.9909468293190002, -0.967984139919281, -0.9927281141281128, -0.9760422706604004, -0.9435782432556152, -0.9706461429595947, -0.9666442275047302, -0.9745851755142212, -0.9583472013473511, -0.9805901050567627, -0.9760699272155762, -0.9754878282546997, -0.9593111276626587, -0.96051424741745, -0.9875031113624573, -0.9745444059371948, -0.9603601694107056, -0.9917795658111572, -1.016481637954712, -0.9884312152862549, -1.0073113441467285, -0.9766584038734436, -0.974114179611206, -1.011623501777649, -1.0079315900802612, -1.0552023649215698, -1.0029739141464233, -1.0396050214767456, -0.971061110496521, -0.9918436408042908, -0.9648165702819824, -0.9433024525642395, -0.9353170990943909, -0.9460000395774841, -0.9017677903175354, -0.923638105392456, -0.9041991233825684, -0.8591073155403137, -0.7553880214691162, -0.6123037338256836]

    rnn_ll_bursty2 = [-2.162060260772705, -1.4354193210601807, -1.4232999086380005, -1.232310175895691, -1.1672475337982178, -1.119668960571289, -1.0743154287338257, -1.0601050853729248, -1.030940294265747, -1.0389140844345093, -1.0289676189422607, -1.0175743103027344, -0.9940139055252075, -1.0097119808197021, -1.0019376277923584, -1.0071055889129639, -1.0048691034317017, -1.0031559467315674, -1.0098166465759277, -1.0293023586273193, -0.9746497869491577, -1.0001932382583618, -1.0160895586013794, -1.0026944875717163, -1.0117640495300293, -1.0480549335479736, -1.0461819171905518, -1.0015513896942139, -1.0296937227249146, -1.039421796798706, -1.0584917068481445, -1.0162752866744995, -1.033453345298767, -1.0202629566192627, -1.0399799346923828, -1.010057806968689, -1.0204545259475708, -0.9648260474205017, -1.0008125305175781, -0.980960488319397, -1.0132269859313965, -0.9592337608337402, -1.0141135454177856, -0.9517045617103577, -1.0184388160705566, -0.9496747255325317, -1.0909936428070068, -0.9258747696876526, -1.1817553043365479, -1.0163925886154175, 4.540771961212158, 0.9941006898880005, -0.7657368779182434, -0.45731472969055176, -0.8160547018051147, -0.7582045197486877, -0.8571605086326599, -0.8476215600967407, -0.8984484672546387, -0.9080387353897095, -0.9531686305999756, -0.9413905739784241, -0.9761133193969727, -0.9628421068191528, -0.9316630959510803, -0.9618886709213257, -0.9597868919372559, -0.9703365564346313, -0.9552597403526306, -0.9782586097717285, -0.9747762680053711, -0.9737648963928223, -0.9580999612808228, -0.9597391486167908, -0.9869436025619507, -0.9740078449249268, -0.9598224759101868, -0.9914382100105286, -1.0161916017532349, -0.9881541132926941, -1.0070842504501343, -0.976441502571106, -0.9739274978637695, -1.0114696025848389, -1.0078084468841553, -1.0550707578659058, -1.0028647184371948, -1.039505124092102, -0.9709700345993042, -0.9917572736740112, -0.9647359251976013, -0.9432268142700195, -0.9352414608001709, -0.9459328055381775, -0.9017001390457153, -0.9235745072364807, -0.9041406512260437, -0.8590450882911682, -0.7553345561027527, -0.6122996211051941]
    rnn_ll_non_bursty3 = [-1.2674989700317383, -0.7088778018951416, -0.7900391221046448, -0.6380290389060974, -0.6181867718696594, -0.5288081169128418, -0.5175377726554871, -0.4737875759601593, -0.5377592444419861, -0.5672685503959656, -0.5223391652107239, -0.5199770331382751, -0.5450291037559509, -0.5372273921966553, -0.5504136681556702, -0.5722652673721313, -0.5189151167869568, -0.5388198494911194, -0.5162752270698547, -0.6182715892791748, -0.6317805647850037, -0.6005231738090515, -0.5649139881134033, -0.6325377821922302, -0.5929129123687744, -0.5421217679977417, -0.514777660369873, -0.4949685037136078, -0.577508807182312, -0.5381590127944946, -0.6210611462593079, -0.5076613426208496, -0.5421379208564758, -0.6071042418479919, -0.533047080039978, -0.5168765783309937, -0.5339870452880859, -0.5783747434616089, -0.5275581479072571, -0.5096771717071533, -0.48785364627838135, -0.4705466628074646, -0.5399408936500549, -0.5099928379058838, -0.47780829668045044, -0.550201416015625, -0.5330999493598938, -0.5921881794929504, -0.6078027486801147, -0.5521218776702881, -0.5750917792320251, -0.5902350544929504, -0.532565712928772, -0.5610085725784302, -0.6016727685928345, -0.5450891852378845, -0.5220059752464294, -0.5546733736991882, -0.5565249919891357, -0.5443444848060608, -0.4850817322731018, -0.5725302696228027, -0.4950811564922333, -0.5821757912635803, -0.5879895091056824, -0.5131303071975708, -0.615070641040802, -0.4798508882522583, -0.5757787823677063, -0.5417320132255554, -0.47967424988746643, -0.5298349857330322, -0.6157343983650208, -0.5204769968986511, -0.5101887583732605, -0.537667453289032, -0.5419722199440002, -0.5371782779693604, -0.5662843585014343, -0.5117828249931335, -0.5486470460891724, -0.6339764595031738, -0.5946379899978638, -0.6447869539260864, -0.6053339838981628, -0.5775472521781921, -0.5531191825866699, -0.6595923900604248, -0.5624443888664246, -0.5916541218757629, -0.5118759870529175, -0.5519577860832214, -0.5664625763893127, -0.5920288562774658, -0.49152082204818726, -0.6783906817436218, -0.5359346270561218, -0.5025546550750732, -0.5333302021026611, -0.38167017698287964]
    rnn_ll_bursty3 = [-1.2674773931503296, -0.7088844776153564, -0.7900053262710571, -0.6380545496940613, -0.6181690692901611, -0.5287986397743225, -0.5176295042037964, -0.473702996969223, -0.5377720594406128, -0.5671871304512024, -0.5225525498390198, -0.5200547575950623, -0.5450470447540283, -0.5372940301895142, -0.5502797961235046, -0.5719569325447083, -0.5191314220428467, -0.5393261909484863, -0.5162574052810669, -0.61835116147995, -0.6313521265983582, -0.6004378199577332, -0.5638577342033386, -0.6327562928199768, -0.5916649103164673, -0.5414398908615112, -0.5145938396453857, -0.49359437823295593, -0.578014612197876, -0.542887806892395, -0.623834490776062, -0.5065111517906189, -0.5474100112915039, -0.608693540096283, -0.5326657295227051, -0.5242891311645508, -0.5272159576416016, -0.5730230808258057, -0.528648853302002, -0.5200464129447937, -0.4864168167114258, -0.4823075830936432, -0.5559430122375488, -0.5015975832939148, -0.5149284601211548, -0.4990003705024719, -0.6100702285766602, -0.6036994457244873, -0.7883924841880798, -0.6324856281280518, 4.168789386749268, 0.2844081521034241, -0.6770893335342407, -0.4024193286895752, -0.6499606370925903, -0.5221795439720154, -0.5415438413619995, -0.570417046546936, -0.5771557688713074, -0.5749652981758118, -0.47177955508232117, -0.5758190155029297, -0.48730146884918213, -0.5739664435386658, -0.5805259346961975, -0.5104792714118958, -0.617486834526062, -0.4869902431964874, -0.5715957283973694, -0.5438429117202759, -0.4832479953765869, -0.5297907590866089, -0.6169071793556213, -0.5181573629379272, -0.5089337229728699, -0.5389635562896729, -0.5405763387680054, -0.537243127822876, -0.5653367042541504, -0.5100393295288086, -0.5480112433433533, -0.6345977783203125, -0.5939146876335144, -0.6447418928146362, -0.6049959659576416, -0.5779058337211609, -0.5532793402671814, -0.659524142742157, -0.5620713233947754, -0.5913740992546082, -0.5116147994995117, -0.5519453287124634, -0.5664435029029846, -0.5920177698135376, -0.49157384037971497, -0.678351879119873, -0.5358576774597168, -0.5025247931480408, -0.5333449244499207, -0.3816467225551605]
    rnn_ll_non_bursty4 = [-0.2091672718524933, -0.23539172112941742, -0.32506871223449707, -0.2456856667995453, -0.44053027033805847, -0.32385700941085815, -0.3480248749256134, -0.36564260721206665, -0.39864906668663025, -0.4098392426967621, -0.3407641649246216, -0.31553947925567627, -0.37667304277420044, -0.4872966408729553, -0.4305541217327118, -0.38195762038230896, -0.526767909526825, -0.4200403690338135, -0.3713580071926117, -0.46581408381462097, -0.43763113021850586, -0.3245863616466522, -0.41474413871765137, -0.40994855761528015, -0.29247045516967773, -0.29975593090057373, -0.3784465491771698, -0.36638227105140686, -0.45002225041389465, -0.4423525035381317, -0.4263949394226074, -0.43106478452682495, -0.2974565327167511, -0.3465328812599182, -0.3058770000934601, -0.4069608151912689, -0.5204896330833435, -0.40246593952178955, -0.3261192739009857, -0.29727819561958313, -0.36478888988494873, -0.2809239625930786, -0.3538624048233032, -0.3509809970855713, -0.2773270308971405, -0.4181859791278839, -0.3640967905521393, -0.3398558795452118, -0.3096190094947815, -0.354874849319458, -0.3847476541996002, -0.40927407145500183, -0.31619980931282043, -0.3666294813156128, -0.3088062107563019, -0.37666720151901245, -0.38791215419769287, -0.44267287850379944, -0.3190714120864868, -0.31536349654197693, -0.28408733010292053, -0.3975871801376343, -0.3549819886684418, -0.3353519141674042, -0.4118838906288147, -0.38353484869003296, -0.4337449073791504, -0.34809964895248413, -0.4075985848903656, -0.39325547218322754, -0.4815569221973419, -0.23213790357112885, -0.34484052658081055, -0.4813466966152191, -0.5384815335273743, -0.43296295404434204, -0.36174118518829346, -0.2893470227718353, -0.28039756417274475, -0.34614503383636475, -0.42703479528427124, -0.39886659383773804, -0.375190794467926, -0.3720755875110626, -0.3188915550708771, -0.45652803778648376, -0.39744019508361816, -0.3194970190525055, -0.38977402448654175, -0.2979673743247986, -0.40847402811050415, -0.4051728844642639, -0.3632059693336487, -0.2576432526111603, -0.4395119845867157, -0.32341936230659485, -0.44799697399139404, -0.3672146797180176, -0.4493618905544281, -0.40341269969940186]
    rnn_ll_bursty4 = [-0.2091888189315796, -0.23534780740737915, -0.3250710666179657, -0.24563677608966827, -0.4405447244644165, -0.32394692301750183, -0.3480020761489868, -0.365654319524765, -0.3986510932445526, -0.4097897708415985, -0.34067827463150024, -0.3155217170715332, -0.3764764070510864, -0.48688405752182007, -0.43035638332366943, -0.38190746307373047, -0.5269336104393005, -0.4202447831630707, -0.3715165853500366, -0.46675920486450195, -0.43835100531578064, -0.3240427076816559, -0.41500115394592285, -0.4096396565437317, -0.2938649654388428, -0.29938632249832153, -0.3774164319038391, -0.3627866804599762, -0.44774124026298523, -0.44504109025001526, -0.4243685305118561, -0.43666377663612366, -0.3028619885444641, -0.34629201889038086, -0.32017889618873596, -0.4053592383861542, -0.5277420878410339, -0.41185659170150757, -0.3158595860004425, -0.30543914437294006, -0.31836897134780884, -0.23470091819763184, -0.3221202790737152, -0.402713805437088, -0.2951354384422302, -0.4016570448875427, -0.3964523673057556, -0.40398240089416504, -0.5081165432929993, -0.3435531258583069, 4.061943531036377, -0.0220491960644722, -0.3316289186477661, -0.310026079416275, -0.21162639558315277, -0.2846437692642212, -0.4509860575199127, -0.4362775385379791, -0.3079332113265991, -0.34370410442352295, -0.3068718910217285, -0.4475017488002777, -0.3194534182548523, -0.34978753328323364, -0.40036797523498535, -0.37388840317726135, -0.412777841091156, -0.36375442147254944, -0.4321790635585785, -0.37395671010017395, -0.48989591002464294, -0.2291616052389145, -0.3424642086029053, -0.48144254088401794, -0.5380896925926208, -0.4378463923931122, -0.3644040822982788, -0.28790000081062317, -0.2796361446380615, -0.343330055475235, -0.42921188473701477, -0.400450199842453, -0.3772876262664795, -0.37110137939453125, -0.32067206501960754, -0.4598698019981384, -0.39667150378227234, -0.3209446966648102, -0.3902033865451813, -0.2982378900051117, -0.40816372632980347, -0.40468630194664, -0.36391428112983704, -0.2573500871658325, -0.44018739461898804, -0.3238646686077118, -0.44802573323249817, -0.3671426773071289, -0.44918036460876465, -0.4039807915687561]

    interpret_radar = Interpret(network_saved_path=network_saved_path_radartrained, block_len=100, num_block=1000)

    rnn_ll_non_bursty5, rnn_ll_bursty5 = interpret_radar.likelihood(bit_pos=radar_bit_pos,sigma = 0.5,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True)

    rnn_ll_non_bursty6, rnn_ll_bursty6 = interpret_radar.likelihood(bit_pos=radar_bit_pos,sigma = 1.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True)

    rnn_ll_non_bursty7, rnn_ll_bursty7 = interpret_radar.likelihood(bit_pos=radar_bit_pos,sigma = 2.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True)

    rnn_ll_non_bursty8, rnn_ll_bursty8 = interpret_radar.likelihood(bit_pos=radar_bit_pos,sigma = 5.0,
                                                                  radar_noise_power = 10, is_compute_map=False,
                                                                  is_compute_no_bursty=True)


    plt.figure(1)
    plt.title('likelihood compare on different snr, RNN, No Bursty Noise')
    p1, = plt.plot(rnn_ll_non_bursty1, 'y', label ='AWGN-trained RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_non_bursty2, 'g', label ='AWGN-trained RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_non_bursty3, 'b', label ='AWGN-trained RNN AWGNsigma 2.0')
    #p4, = plt.plot(rnn_ll_non_bursty4, 'r', label ='AWGN-trained RNN AWGN sigma 5.0')

    # p5, = plt.plot(rnn_ll_non_bursty5, 'y-*', label ='radar-trained RNN AWGN sigma 0.5' )
    # p6, = plt.plot(rnn_ll_non_bursty6, 'g-*', label ='radar-trained RNN AWGN sigma 1.0')
    # p7, = plt.plot(rnn_ll_non_bursty7, 'b-*', label ='radar-trained RNN AWGNsigma 2.0')
    # p8, = plt.plot(rnn_ll_non_bursty8, 'r-*', label ='radar-trained RNN AWGN sigma 5.0')

    p5, = plt.plot(rnn_ll_non_bursty5, 'y-*', label ='t3-trained RNN AWGN sigma 0.5' )
    p6, = plt.plot(rnn_ll_non_bursty6, 'g-*', label ='t3-trained RNN AWGN sigma 1.0')
    p7, = plt.plot(rnn_ll_non_bursty7, 'b-*', label ='t3-trained RNN AWGNsigma 2.0')
    #p8, = plt.plot(rnn_ll_non_bursty8, 'r-*', label ='radar-trained RNN AWGN sigma 5.0')
    #plt.legend(handles = [p2, p4, p6, p8])
    plt.legend(handles = [p1, p2, p3, p5, p6, p7])

    plt.figure(2)
    plt.title('likelihood compare on different snr, RNN, with Bursty Noise')
    p1, = plt.plot(rnn_ll_bursty1, 'y', label ='AWGN-trained RNN AWGN sigma 0.5' )
    p2, = plt.plot(rnn_ll_bursty2, 'g', label ='AWGN-trained RNN AWGN sigma 1.0')
    p3, = plt.plot(rnn_ll_bursty3, 'b', label ='AWGN-trained RNN AWGNsigma 2.0')
    #p4, = plt.plot(rnn_ll_bursty4, 'r', label ='AWGN-trained RNN AWGN sigma 5.0')

    # p5, = plt.plot(rnn_ll_bursty5, 'y-*', label ='radar-trained RNN AWGN sigma 0.5' )
    # p6, = plt.plot(rnn_ll_bursty6, 'g-*', label ='radar-trained RNN AWGN sigma 1.0')
    # p7, = plt.plot(rnn_ll_bursty7, 'b-*', label ='radar-trained RNN AWGNsigma 2.0')
    # p8, = plt.plot(rnn_ll_bursty8, 'r-*', label ='radar-trained RNN AWGN sigma 5.0')

    p5, = plt.plot(rnn_ll_bursty5, 'y-*', label ='t3-trained RNN AWGN sigma 0.5' )
    p6, = plt.plot(rnn_ll_bursty6, 'g-*', label ='t3-trained RNN AWGN sigma 1.0')
    p7, = plt.plot(rnn_ll_bursty7, 'b-*', label ='t3-trained RNN AWGNsigma 2.0')
    #p8, = plt.plot(rnn_ll_bursty8, 'r-*', label ='t3-trained RNN AWGN sigma 5.0')
    plt.legend(handles = [p1, p2, p3, p5, p6, p7])
    #plt.legend(handles = [p2, p4, p6, p8])
    plt.show()


if __name__ == '__main__':
    # User Case 1, likelihood on bursty noise and AWGN only. Compare RNN and BCJR's output.
    #likelihood_snr_range()

    # User Case 2, likelihood on Radar-trained model and AWGN-trained model. Compare the output scale.
    likelihood_radartrained_vs_awgntrained()
    print haha





    # ber_vs_power_res = []
    # for this_noise_power in radar_noise_power:
    #     this_ber = interpret.ber(bit_pos=radar_bit_pos, noise_power=this_noise_power)
    #     ber_vs_power_res.append(this_ber)

    # usage 2:radar noise power vs likelihood, with fixed noise pos = 50
    # likelihood_vs_power_res = []
    # for this_noise_power in radar_noise_power:
    #     this_likelihood = interpret.likelihood(bit_pos=radar_bit_pos, noise_power=this_noise_power,
    #                                            is_compute_no_bursty = False ,is_compute_map = False)
    #     likelihood_vs_power_res.append(this_likelihood)
    # map_ll_bursty = np.stack(np.array(map_likelihood_list), axis=0)
    # map_ll_bursty = np.mean(map_ll_bursty, axis=0).T.tolist()


    # usage 3:radar noise position vs ber, with fixed noise power 0dB
    this_noise_power = 1.0
    likelihood_vs_noise_pos_res = []
    for radar_bit_pos in radar_noise_power:
        this_ber = interpret.ber(bit_pos=radar_bit_pos, noise_power=this_noise_power)
        likelihood_vs_noise_pos_res.append(this_ber)

    # usage 4:radar noise position vs likelihood, with fixed noise power 0dB
    radar_noise_pos_list = [10*item for item in range(10)]
    ber_vs_noise_pos_res = []
    for radar_bit_pos in radar_noise_power:
        this_likelihood = interpret.likelihood(bit_pos=radar_bit_pos, noise_power=this_noise_power)
        ber_vs_noise_pos_res.append(this_likelihood)

    ###############################################
    # Plot Graph
    ###############################################








