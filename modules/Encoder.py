import cv2
import numpy as np
from scipy.fft import dct

class Encoder:
    """
    This class provides functionality for encoding an image using various techniques.
    """

    def __init__(self, img, downsampling_rate=422, interpolation=cv2.INTER_LINEAR, block_size=None):
        """
        Initializes an encoder object with provided image and parameters.

        Parameters:
            img (numpy.ndarray): Input image in RGB format.
            downsampling_rate (int): Downsampling rate for chroma subsampling. Default is 422.
            interpolation (int): Interpolation method for resizing during downsampling. Default is cv2.INTER_LINEAR.
            block_size (int or None): Size of the blocks for DCT calculation. If None, DCT is applied to the entire channel without blocking. Default is None.

        Attributes:
            img (numpy.ndarray): Input image in RGB format.
            block_size (int or None): Size of the blocks for DCT calculation.
            downsampling_rate (int): Downsampling rate for chroma subsampling.
            interpolation (int): Interpolation method for resizing during downsampling.
            R (numpy.ndarray): Red channel of the image.
            G (numpy.ndarray): Green channel of the image.
            B (numpy.ndarray): Blue channel of the image.
            rows (int): Number of rows in the image.
            cols (int): Number of columns in the image.
            Y (numpy.ndarray): Luma (Y) component in YCbCr color space.
            Cb (numpy.ndarray): Chroma blue (Cb) component in YCbCr color space.
            Cr (numpy.ndarray): Chroma red (Cr) component in YCbCr color space.
            Y_d (numpy.ndarray): Downsampled luma (Y) component.
            Cb_d (numpy.ndarray): Downsampled chroma blue (Cb) component.
            Cr_d (numpy.ndarray): Downsampled chroma red (Cr) component.
            Y_DCT (numpy.ndarray): DCT coefficients of downsampled luma (Y) component.
            Cb_DCT (numpy.ndarray): DCT coefficients of downsampled chroma blue (Cb) component.
            Cr_DCT (numpy.ndarray): DCT coefficients of downsampled chroma red (Cr) component.
        """
        self.img = img
        self.block_size = block_size
        self.downsampling_rate = downsampling_rate
        self.interpolation = interpolation
        self.R, self.G, self.B = self.split_rgb(img)
        self.rows, self.cols = self.R.shape
        self.R, self.G, self.B = self.apply_padding(self.R), self.apply_padding(self.G), self.apply_padding(self.B)
        self.Y, self.Cb, self.Cr = self.convert_to_ycbcr()
        self.Y_d, self.Cb_d, self.Cr_d = self.downsample_ycbcr(downsampling_rate=downsampling_rate, interpolation=interpolation)
        self.Y_DCT, self.Cb_DCT, self.Cr_DCT = self.calculate_dct(self.Y_d), self.calculate_dct(self.Cb_d), self.calculate_dct(self.Cr_d)

    def split_rgb(self, img):
        """
        Splits the input RGB image into its red, green, and blue channels.

        Parameters:
            img (numpy.ndarray): Input RGB image.

        Returns:
            tuple: Red, green, and blue channels as numpy arrays.
        """
        r = img.image[:, :, 0]
        g = img.image[:, :, 1]
        b = img.image[:, :, 2]
        return r, g, b

    def apply_padding(self, channel):
        """
        Applies padding to the given channel to ensure its dimensions are multiples of 32.

        Parameters:
            channel (numpy.ndarray): Input channel.

        Returns:
            numpy.ndarray: Padded channel.
        """
        channel = self.apply_vertical_padding(channel)
        channel = self.apply_horizontal_padding(channel)
        return channel

    def apply_vertical_padding(self, channel):
        """
        Applies vertical padding to the given channel to ensure its dimensions are multiples of 32.

        Parameters:
            channel (numpy.ndarray): Input channel.

        Returns:
            numpy.ndarray: Vertically padded channel.
        """
        line_n = channel.shape[0]
        vertical_padding = 32 - (line_n % 32)
        vertical_array = np.repeat(channel[-1:, :], vertical_padding, 0)
        channel = np.vstack((channel, vertical_array))
        return channel

    def apply_horizontal_padding(self, channel):
        """
        Applies horizontal padding to the given channel to ensure its dimensions are multiples of 32.

        Parameters:
            channel (numpy.ndarray): Input channel.

        Returns:
            numpy.ndarray: Horizontally padded channel.
        """
        column_n = channel.shape[1]
        horizontal_padding = 32 - (column_n % 32)
        horizontal_array = np.repeat(channel[:, -1:], horizontal_padding, 1)
        channel = np.hstack((channel, horizontal_array))
        return channel

    def convert_to_ycbcr(self):
        """
        Converts the RGB image to YCbCr color space.

        Returns:
            tuple: Luma (Y), chroma blue (Cb), and chroma red (Cr) components in YCbCr color space.
        """
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
    
    def downsample_ycbcr(self, downsampling_rate=422, interpolation=cv2.INTER_LINEAR):
        """
        Downsamples the chroma blue (Cb) and chroma red (Cr) components based on the specified downsampling rate.

        Parameters:
            downsampling_rate (int): Downsampling rate for chroma subsampling.
            interpolation (int): Interpolation method for resizing.

        Returns:
            tuple: Downsampled luma (Y), chroma blue (Cb), and chroma red (Cr) components.
        """
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
        """
        Calculates the Discrete Cosine Transform (DCT) coefficients of the given channel.

        If block_size is None, the entire channel is transformed using DCT.
        If block_size is specified, the channel is divided into blocks of size block_size x block_size, and DCT is applied to each block.

        Parameters:
            channel (numpy.ndarray): Input channel for which DCT coefficients are calculated.

        Returns:
            numpy.ndarray: DCT coefficients of the input channel.
        """
        if self.block_size is None:
            return dct(dct(channel, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel.shape
        channel_dct = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.block_size):
            for j in range(0, channel_shape[1], self.block_size):
                channel_dct[i:i+self.block_size, j:j+self.block_size] = dct(dct(channel[i:i+self.block_size, j:j+self.block_size], norm='ortho').T, norm='ortho').T
        
        return channel_dct