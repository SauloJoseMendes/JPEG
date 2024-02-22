import numpy as np
from modules.Encoder import Encoder
import cv2
from scipy.fftpack import idct

class Decoder:
    def __init__(self, encoded_img):
        self.encoded_img = encoded_img
        self.Y_d, self.Cb_d, self.Cr_d = self.calculate_idct(self.encoded_img.Y_DCT), self.calculate_idct(self.encoded_img.Cb_DCT), self.calculate_idct(self.encoded_img.Cr_DCT)
        self.Y_up, self.Cb_up, self.Cr_up = self.upsample_ycbcr()
        self.R, self.G, self.B = self.convert_to_rgb()
        self.R, self.G, self.B = self.remove_padding(self.R), self.remove_padding(self.G), self.remove_padding(self.B)
        self.RGB = self.join_rgb(self.R, self.G, self.B)


    def convert_to_rgb(self):
        conversion_matrix = np.array([[1., 0., 1.402],
                                      [1., -0.344136, -0.714136],
                                      [1., 1.772, 0.]])

        ycbcr = np.stack((self.Y_up, self.Cb_up, self.Cr_up))
        rgb = np.dot(conversion_matrix, ycbcr.reshape(3, -1) - np.array([[0], [128], [128]]))
        rgb = np.round(rgb)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.reshape(ycbcr.shape)

        return rgb.astype(np.uint8)

    def join_rgb(self, r, g, b):
        nl, nc = r.shape
        img_rec = np.zeros((nl, nc, 3), dtype=np.uint8)
        img_rec[:, :, 0] = r
        img_rec[:, :, 1] = g
        img_rec[:, :, 2] = b
        return img_rec

    def remove_padding(self, channel):
        return channel[: self.encoded_img.rows, : self.encoded_img.cols]
    
    def upsample_ycbcr(self):
        # Get the dimensions of the original Y channel
        original_height, original_width = self.Y_d.shape
        interpolation = self.encoded_img.interpolation
        # Resize the Cb and Cr channels to match the original Y channel dimensions
        Cb_up = cv2.resize(self.Cb_d, (original_width, original_height), interpolation=interpolation)
        Cr_up = cv2.resize(self.Cr_d, (original_width, original_height), interpolation=interpolation)
        return self.Y_d, Cb_up, Cr_up
    
    def calculate_idct(self, channel_dct):
        if self.encoded_img.block_size is None:
            return idct(idct(channel_dct, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel_dct.shape
        channel = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.encoded_img.block_size):
            for j in range(0, channel_shape[1], self.encoded_img.block_size):
                channel[i:i+self.encoded_img.block_size, j:j+self.encoded_img.block_size] = idct(idct(channel_dct[i:i+self.encoded_img.block_size, j:j+self.encoded_img.block_size], norm='ortho').T, norm='ortho').T
        
        return channel
