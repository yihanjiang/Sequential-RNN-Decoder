__author__ = 'yihanjiang'
'''
This module is for future speedup of Commpy.
'''
import numpy as np



def encode_turbo():


    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[7, 5]])
    trellis1 = cc.Trellis(M, generator_matrix,feedback=7)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=7)# Create trellis data structure

    interleaver = RandInterlv.RandInterlv(k,0)
    p_array = interleaver.p_array


def decode_turbo_multiprocess(X_feed):
    import multiprocessing as mp



    pass

if __name__ == '__main__':
    decode_turbo_multiprocess()