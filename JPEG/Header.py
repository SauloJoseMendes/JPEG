import cv2
import numpy as np
from scipy.fft import dct

class Header:
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                              [-0.168736, -0.331264, 0.5],
                              [0.5, -0.418688, -0.081312]])

    conversion_matrix_inversed = np.linalg.inv(conversion_matrix)

    YCbCr_offset = np.array([[0], [128], [128]])

    Q_Y = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

    Q_CbCr = np.array([[17,18,24,47,99,99,99,99],
                    [18,21,26,66,99,99,99,99],
                    [24,26,56,99,99,99,99,99],
                    [47,66,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99]])

    def __init__(self, downsampling_rate=422, interpolation=cv2.INTER_LINEAR, block_size=None, quality_factor = 75):
        self.downsampling_rate = downsampling_rate
        self.interpolation = interpolation
        self.block_size = block_size
        self.rows = 0 
        self.columns = 0
        self.quality_factor = quality_factor
