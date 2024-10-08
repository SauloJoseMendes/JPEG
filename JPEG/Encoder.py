import cv2
import numpy as np
from scipy.fft import dct
from JPEG.Header import Header

class Encoder:
    """
    This class provides functionality for encoding an image using various techniques.
    """

    def __init__(self, img, header = Header()):
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
        self.header = header
        self.R, self.G, self.B = self.split_rgb(img)
        self.header.rows, self.header.columns = self.R.shape
        self.R, self.G, self.B = self.apply_padding(self.R), self.apply_padding(self.G), self.apply_padding(self.B)
        self.Y, self.Cb, self.Cr = self.convert_to_ycbcr()
        self.Y_d, self.Cb_d, self.Cr_d = self.downsample_ycbcr()
        self.Y_DCT, self.Cb_DCT, self.Cr_DCT = self.calculate_dct(self.Y_d), self.calculate_dct(self.Cb_d), self.calculate_dct(self.Cr_d)
        self.Y_Q, self.Cb_Q, self.Cr_Q = self.quantize(self.Y_DCT,is_y=1), self.quantize(self.Cb_DCT,is_y=0), self.quantize(self.Cr_DCT,is_y=0)
        self.Y_DPCM, self.Cb_DPCM, self.Cr_DPCM= self.apply_DPCM(self.Y_Q), self.apply_DPCM(self.Cb_Q), self.apply_DPCM(self.Cr_Q)

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

        rgb = np.stack((self.R, self.G, self.B))

        ycbcr = np.dot(self.header.conversion_matrix, rgb.reshape(3, -1)) + self.header.YCbCr_offset
        ycbcr = np.round(ycbcr)
        ycbcr = np.clip(ycbcr, 0, 255)
        ycbcr = ycbcr.reshape(rgb.shape)

        return ycbcr.astype(np.uint8)
    
    def downsample_ycbcr(self):
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
        if self.header.downsampling_rate == 422:
            scale_factor_h = 0.5
            scale_factor_v = 1
        if self.header.downsampling_rate == 420:
            scale_factor_h = 0.5
            scale_factor_v = 0.5
        Cb_new_size = (int(self.Cb.shape[1] * scale_factor_h), int(self.Cb.shape[0] * scale_factor_v))
        Cr_new_size = (int(self.Cr.shape[1] * scale_factor_h), int(self.Cr.shape[0] * scale_factor_v))
        Cb_d = cv2.resize(self.Cb, Cb_new_size, interpolation=self.header.interpolation)
        Cr_d = cv2.resize(self.Cr, Cr_new_size, interpolation=self.header.interpolation)
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
        if self.header.block_size is None:
            return dct(dct(channel, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel.shape
        channel_dct = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.header.block_size):
            for j in range(0, channel_shape[1], self.header.block_size):
                channel_dct[i:i+self.header.block_size, j:j+self.header.block_size] = dct(dct(channel[i:i+self.header.block_size, j:j+self.header.block_size], norm='ortho').T, norm='ortho').T
        
        return channel_dct
    
    def quantize(self, channel, is_y):
        """
        Quantize the given channel using the provided quantization matrix.

        Args:
            channel (numpy.ndarray): The input channel to be quantized.
            is_y (bool): A boolean indicating whether the channel is luminance (Y) or chrominance (CbCr).

        Returns:
            numpy.ndarray: The quantized channel.

        Notes:
            The quantization is performed according to the JPEG standard.
            For luminance (Y) channel, the quantization matrix is fetched from self.header.Q_Y.
            For chrominance (CbCr) channels, the quantization matrix is fetched from self.header.Q_CbCr.
            The quantization step is adjusted based on the quality factor stored in self.header.quality_factor.
            If quality_factor >= 50, sf is calculated as (100 - quality_factor) / 50.
            If quality_factor < 50, sf is calculated as 50 / quality_factor.
            Quantization scale factor (qsf) is obtained by multiplying the quantization matrix with sf,
            rounded and clipped to the range [1, 255].
            The channel is then divided element-wise by qsf and rounded to the nearest integer.
            If sf equals 0, indicating lossless compression, the channel remains unchanged.
        """
        channel_shape = channel.shape
        Q_channel = np.zeros(channel_shape)

        if is_y:
            quant_matrix = self.header.Q_Y
        else:
            quant_matrix = self.header.Q_CbCr

        if self.header.quality_factor >= 50:
            sf = (100 - self.header.quality_factor) / 50
        else:
            sf = 50 / self.header.quality_factor
        
        if sf != 0:
            qsf = np.round(quant_matrix * sf)
            qsf = np.clip(qsf, 1, 255)

            for i in range(0, channel_shape[0], 8):
                for j in range(0, channel_shape[1], 8):
                    Q_channel[i:i+8, j:j+8] = channel[i:i+8, j:j+8] / qsf
            Q_channel = np.round(Q_channel)
        else:
            Q_channel = np.round(channel)

        return Q_channel.astype(np.int16)

    def apply_DPCM(self, channel):
        """
        Apply Differential Pulse Code Modulation (DPCM) to the given channel.

        Args:
            channel (numpy.ndarray): The input channel to which DPCM is applied.

        Returns:
            numpy.ndarray: The channel after DPCM encoding.

        Notes:
            Differential Pulse Code Modulation (DPCM) is a form of lossless data compression.
            DPCM predicts the value of each sample by using the difference between the
            current and the previous sample. This difference, or error signal, is then
            quantized and encoded.
            In this implementation, DPCM is applied on an 8x8 block basis.
            For each 8x8 block in the channel, the difference between each element and
            its neighboring element (previous element in the same row or column) is computed
            and stored in the same location.
            The prediction process starts from the bottom-right corner of the channel and
            iterates towards the top-left corner. The bottom-right corner is skipped since
            it has no neighboring elements.
        """
        channel_shape = channel.shape
        dpcm = np.copy(channel)

        for r in range(channel_shape[0]-8, -8, -8):
            for c in range(channel_shape[1]-8, -8, -8):
                if r == 0 and c == 0:
                    continue
                if c == 0:
                    dpcm[r, 0] = dpcm[r, 0] - dpcm[r-8, channel_shape[1]-8]
                else:
                    dpcm[r, c] = dpcm[r, c] - dpcm[r, c-8]

        return dpcm
