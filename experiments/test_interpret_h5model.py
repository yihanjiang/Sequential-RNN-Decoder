__author__ = 'yihanjiang'

import numpy as np
import matplotlib.pyplot as plt

def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxweight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()

    plt.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')

    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, -w/maxweight),
                      'black')
    if reenable:
        plt.ion()


def level1():
    import h5py
    #hdf5_file_name = './model_zoo/radar_model_end2end/0919radar_end2end_ttbl_0.326416625019_snr_0.h5'
    hdf5_file_name = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_9.h5'
    #hdf5_file_name = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_9.h5'
    file    = h5py.File(hdf5_file_name, 'r')

    print 'running on layer 1'

    bias_matrix   = file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['bias:0'][:]
    recur_matrix = file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['recurrent_kernel:0'][:, :]
    kernel_matrix = file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['kernel:0'][:, :]
    print file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['recurrent_kernel:0']
    print file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['kernel:0']

    b2_input    = bias_matrix[ :200].T
    b2_forget   = bias_matrix[ 200:400].T
    b2_mem_cell = bias_matrix[ 400:600].T
    b2_output   = bias_matrix[ 600:].T

    print 'Mean of Bias Layer'
    print np.mean(b2_input)
    print np.mean(b2_forget)
    print np.mean(b2_mem_cell)
    print np.mean(b2_output)
    print np.mean(abs(b2_input))
    print np.mean(abs(b2_forget))
    print np.mean(abs(b2_mem_cell))
    print np.mean(abs(b2_output))

    k2_input    = kernel_matrix[:, :200]
    k2_forget   = kernel_matrix[:, 200:400]
    k2_mem_cell = kernel_matrix[:, 400:600]
    k2_output   = kernel_matrix[:, 600:]

    r2_input    = recur_matrix[:, :200]
    r2_forget   = recur_matrix[:, 200:400]
    r2_mem_cell = recur_matrix[:, 400:600]
    r2_output   = recur_matrix[:, 600:]

    from numpy import linalg as LA
    print 'Kernel Matrix Eigen Values, max/mean'
    w, v = LA.eig(np.dot(k2_input, k2_input.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_forget, k2_forget.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_mem_cell, k2_mem_cell.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_output, k2_output.T))
    print max(w), np.mean(w)

    print 'Recurrent Matrix Eigen Values, max/mean'
    w, v = LA.eig(np.dot(r2_input, r2_input.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(r2_forget,r2_forget.T))
    print max(w), np.mean(w)


    w, v = LA.eig(np.dot(r2_mem_cell, r2_mem_cell.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(r2_output, r2_output.T))
    print max(w), np.mean(w)

def level2():
    import h5py
    #hdf5_file_name = './model_zoo/radar_model_end2end/0919radar_end2end_ttbl_0.326416625019_snr_0.h5'
    hdf5_file_name = './model_zoo/awgn_model_end2end/yihan_clean_ttbl_0.870905022927_snr_9.h5'
    #hdf5_file_name = './model_zoo/tdist_v3_model_end2end/tdist_end2end_ttbl_0.440818870589_snr_9.h5'
    file    = h5py.File(hdf5_file_name, 'r')

    print 'running on layer 2'

    print file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2'].keys()

    bias_matrix   = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['bias:0'][:]
    recur_matrix  = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['recurrent_kernel:0'][:, :]
    kernel_matrix = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['kernel:0'][:, :]


    # recur_matrix = file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['recurrent_kernel:0'][:, :]
    # kernel_matrix = file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['kernel:0'][:, :]
    # print file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['recurrent_kernel:0']
    # print file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1']['kernel:0']

    b2_input    = bias_matrix[ :200].T
    b2_forget   = bias_matrix[ 200:400].T
    b2_mem_cell = bias_matrix[ 400:600].T
    b2_output   = bias_matrix[ 600:].T

    print 'Mean of Bias Layer'
    print np.mean(b2_input)
    print np.mean(b2_forget)
    print np.mean(b2_mem_cell)
    print np.mean(b2_output)

    k2_input    = kernel_matrix[:, :200].T
    k2_forget   = kernel_matrix[:, 200:400].T
    k2_mem_cell = kernel_matrix[:, 400:600].T
    k2_output   = kernel_matrix[:, 600:].T

    r2_input    = recur_matrix[:, :200]
    r2_forget   = recur_matrix[:, 200:400]
    r2_mem_cell = recur_matrix[:, 400:600]
    r2_output   = recur_matrix[:, 600:]

    from numpy import linalg as LA
    print 'Kernel Matrix Eigen Values, max/mean'
    w, v = LA.eig(np.dot(k2_input, k2_input.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_forget, k2_forget.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_mem_cell, k2_mem_cell.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(k2_output, k2_output.T))
    print max(w), np.mean(w)

    print 'Recurrent Matrix Eigen Values, max/mean'
    w, v = LA.eig(np.dot(r2_input, r2_input.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(r2_forget,r2_forget.T))
    print max(w), np.mean(w)


    w, v = LA.eig(np.dot(r2_mem_cell, r2_mem_cell.T))
    print max(w), np.mean(w)

    w, v = LA.eig(np.dot(r2_output, r2_output.T))
    print max(w), np.mean(w)


    # print np.mean(r2_input), np.mean(r2_forget),np.mean(r2_mem_cell),np.mean(r2_output)
    #
    # print np.mean(recur_matrix), np.var(recur_matrix)

    # hinton(recur_matrix[:200, :200])
    # plt.title('Example Hinton diagram - 200x200 random normal')
    # plt.show()

if __name__ == "__main__":
    # level2()
    # level1()
    import h5py
    hdf5_file_name = './model_zoo/bcjr_train_0925/bcjr_traindefault_200_simple-rnn_sd_1000.570911494253_2.h5'
    file    = h5py.File(hdf5_file_name, 'r')

    print 'running on layer 2'

    print file.keys()

    print file['bidirectional_1']['bidirectional_1']['forward_bidirectional_1'].keys()

    bias_matrix   = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['bias:0'][:]
    recur_matrix  = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['recurrent_kernel:0'][:, :]
    kernel_matrix = file['bidirectional_2']['bidirectional_2']['forward_bidirectional_2']['kernel:0'][:, :]
