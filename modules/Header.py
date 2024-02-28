import cv2
import numpy as np
from scipy.fft import dct

class Header:
    def __init__(self, downsampling_rate=422, interpolation=cv2.INTER_LINEAR, block_size=None):
        self.downsampling_rate = downsampling_rate
        self.interpolation = interpolation
        self.block_size = block_size
        self.rows = 0 
        self.columns = 0