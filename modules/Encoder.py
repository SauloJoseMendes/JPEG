import numpy as np
import cv2
from scipy.fftpack import dct
class Encoder:
    def __init__(self, img, downsampling_rate = 422, interpolation = cv2.INTER_LINEAR, block_size = None):
        self.img = img
        self.block_size = block_size
        self.downsampling_rate = downsampling_rate
        self.interpolation = interpolation
        self.R, self.G, self.B = self.split_rgb(img)
        self.rows, self.cols = self.R.shape
        self.R, self.G, self.B = self.apply_padding(self.R), self.apply_padding(self.G), self.apply_padding(self.B)
        self.Y, self.Cb, self.Cr = self.convert_to_ycbcr()
        self.Y_d, self.Cb_d, self.Cr_d = self.downsample_ycbcr(downsampling_rate = downsampling_rate, interpolation = interpolation)
        self.Y_DCT, self.Cb_DCT, self.Cr_DCT = self.calculate_dct(self.Y_d), self.calculate_dct(self.Cb_d), self.calculate_dct(self.Cr_d)

    def split_rgb(self, img):
        r = img.image[:, :, 0]
        g = img.image[:, :, 1]
        b = img.image[:, :, 2]
        return r, g, b

    def apply_padding(self, channel):
        channel = self.apply_vertical_padding(channel)
        channel = self.apply_horizontal_padding(channel)
        return channel

    def apply_vertical_padding(self, channel):
        # ........ vertical padding (add rows) ..........
        line_n = channel.shape[0]
        vertical_padding = 32 - (line_n % 32)
        vertical_array = np.repeat(channel[-1:, :], vertical_padding, 0)
        channel = np.vstack((channel, vertical_array))
        return channel

    def apply_horizontal_padding(self, channel):
        # ........ horizontal padding (add colums) ..........
        column_n = channel.shape[1]
        horizontal_padding = 32 - (column_n % 32)
        horizontal_array = np.repeat(channel[:, -1:], horizontal_padding, 1)
        channel = np.hstack((channel, horizontal_array))
        return channel

    def convert_to_ycbcr(self):
        conversion_matrix = np.array([[1., 0., 1.402],
                                      [1., -0.344136, -0.714136],
                                      [1., 1.772, 0.]])
        
        conversion_matrix_inversed = np.linalg.inv(conversion_matrix)

        rgb = np.stack((self.R, self.G, self.B))

        ycbcr = np.dot(conversion_matrix_inversed, rgb.reshape(3, -1)) + np.array([[0], [128], [128]])
        ycbcr = np.round(ycbcr)
        ycbcr = np.clip(ycbcr, 0, 255)
        ycbcr = ycbcr.reshape(rgb.shape)

        return ycbcr.astype(np.uint8)
    
    def downsample_ycbcr(self, downsampling_rate = 422, interpolation = cv2.INTER_LINEAR):
        scale_factor_h = 1
        scale_factor_v = 1
        if downsampling_rate == 422:
            scale_factor_h = 0.5
            scale_factor_v = 1
        if downsampling_rate == 420:
            scale_factor_h = 0.5
            scale_factor_v = 0.5
        Cb_new_size = (int(self.Cb.shape[1] * scale_factor_h), int(self.Cb.shape[0] * scale_factor_v))
        Cr_new_size = (int(self.Cr.shape[1] * scale_factor_h), int(self.Cr.shape[0] * scale_factor_v))
        Cb_d = cv2.resize(self.Cb, Cb_new_size, interpolation=interpolation)
        Cr_d = cv2.resize(self.Cr, Cr_new_size, interpolation=interpolation)
        return self.Y, Cb_d, Cr_d
    
    def calculate_dct(self, channel):
        if self.block_size is None:
            return dct(dct(channel, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel.shape
        channel_dct = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.block_size):
            for j in range(0, channel_shape[1], self.block_size):
                channel_dct[i:i+self.block_size, j:j+self.block_size] = dct(dct(channel[i:i+self.block_size, j:j+self.block_size], norm='ortho').T, norm='ortho').T
        
        return channel_dct