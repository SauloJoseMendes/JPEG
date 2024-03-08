import numpy as np
import cv2
from scipy.fftpack import idct

class Decoder:
    """
    This class provides functionality for decoding an encoded image back to its original format.

    Attributes:
        encoded_img (Encoder): An instance of the Encoder class representing the encoded image.
        Y_d (numpy.ndarray): Downsampled luma (Y) component obtained from the encoded image's DCT coefficients.
        Cb_d (numpy.ndarray): Downsampled chroma blue (Cb) component obtained from the encoded image's DCT coefficients.
        Cr_d (numpy.ndarray): Downsampled chroma red (Cr) component obtained from the encoded image's DCT coefficients.
        Y_up (numpy.ndarray): Upsampled luma (Y) component.
        Cb_up (numpy.ndarray): Upsampled chroma blue (Cb) component.
        Cr_up (numpy.ndarray): Upsampled chroma red (Cr) component.
        R (numpy.ndarray): Reconstructed red channel of the image.
        G (numpy.ndarray): Reconstructed green channel of the image.
        B (numpy.ndarray): Reconstructed blue channel of the image.
        RGB (numpy.ndarray): Reconstructed RGB image.
    """
    def __init__(self, encoded_img):
        """
        Initializes a Decoder object with the provided encoded image.

        Parameters:
            encoded_img (Encoder): An instance of the Encoder class representing the encoded image.
        """
        self.encoded_img = encoded_img
        self.header = encoded_img.header
        self.Y_Q, self.Cb_Q, self.Cr_Q= self.apply_iDPCM(self.encoded_img.Y_DPCM), self.apply_iDPCM(self.encoded_img.Cb_DPCM), self.apply_iDPCM(self.encoded_img.Cr_DPCM)
        self.Y_DCT, self.Cb_DCT, self.Cr_DCT = self.inv_quantize(self.Y_Q,is_y=1), self.inv_quantize(self.Cb_Q,is_y=0), self.inv_quantize(self.Cr_Q,is_y=0)
        self.Y_d, self.Cb_d, self.Cr_d = self.calculate_idct(self.Y_DCT), self.calculate_idct(self.Cb_DCT), self.calculate_idct(self.Cr_DCT)
        self.Y_up, self.Cb_up, self.Cr_up = self.upsample_ycbcr()
        self.R, self.G, self.B = self.convert_to_rgb()
        self.R, self.G, self.B = self.remove_padding(self.R), self.remove_padding(self.G), self.remove_padding(self.B)
        self.RGB = self.join_rgb(self.R, self.G, self.B)


    def convert_to_rgb(self):
        """
        Converts the YCbCr components back to RGB color space.

        Returns:
            numpy.ndarray: Reconstructed RGB image.
        """

        ycbcr = np.stack((self.Y_up, self.Cb_up, self.Cr_up))
        rgb = np.dot(self.header.conversion_matrix_inversed, ycbcr.reshape(3, -1) - self.header.YCbCr_offset)
        rgb = np.round(rgb)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.reshape(ycbcr.shape)

        return rgb.astype(np.uint8)

    def join_rgb(self, r, g, b):
        """
        Combines the individual red, green, and blue channels into an RGB image.

        Parameters:
            r (numpy.ndarray): Reconstructed red channel.
            g (numpy.ndarray): Reconstructed green channel.
            b (numpy.ndarray): Reconstructed blue channel.

        Returns:
            numpy.ndarray: Reconstructed RGB image.
        """
        nl, nc = r.shape
        img_rec = np.zeros((nl, nc, 3), dtype=np.uint8)
        img_rec[:, :, 0] = r
        img_rec[:, :, 1] = g
        img_rec[:, :, 2] = b
        return img_rec

    def remove_padding(self, channel):
        """
        Removes padding from the reconstructed channel.

        Parameters:
            channel (numpy.ndarray): Reconstructed channel.

        Returns:
            numpy.ndarray: Channel without padding.
        """
        return channel[: self.header.rows, : self.header.columns]
    
    def upsample_ycbcr(self):
        """
        Upsamples the chroma blue (Cb) and chroma red (Cr) components to match the dimensions of the luma (Y) component.

        Returns:
            tuple: Upsampled luma (Y), chroma blue (Cb), and chroma red (Cr) components.
        """
        # Get the dimensions of the original Y channel
        original_height, original_width = self.Y_d.shape
        interpolation = self.header.interpolation
        # Resize the Cb and Cr channels to match the original Y channel dimensions
        Cb_up = cv2.resize(self.Cb_d, (original_width, original_height), interpolation=interpolation)
        Cr_up = cv2.resize(self.Cr_d, (original_width, original_height), interpolation=interpolation)
        return self.Y_d, Cb_up, Cr_up
    
    def calculate_idct(self, channel_dct):
        """
        Calculates the Inverse Discrete Cosine Transform (IDCT) of the given channel.

        If block_size is None, IDCT is applied to the entire channel.
        If block_size is specified, IDCT is applied to each block separately.

        Parameters:
            channel_dct (numpy.ndarray): DCT coefficients of the channel.

        Returns:
            numpy.ndarray: Reconstructed channel.
        """
        if self.header.block_size is None:
            return idct(idct(channel_dct, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel_dct.shape
        channel = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.header.block_size):
            for j in range(0, channel_shape[1], self.header.block_size):
                channel[i:i+self.header.block_size, j:j+self.header.block_size] = idct(idct(channel_dct[i:i+self.header.block_size, j:j+self.header.block_size], norm='ortho').T, norm='ortho').T
        
        return channel
    
    def inv_quantize(self, channel, is_y):
        """
        Inverse quantize the given channel using the provided quantization matrix.

        Args:
            channel (numpy.ndarray): The input quantized channel.
            is_y (bool): A boolean indicating whether the channel is luminance (Y) or chrominance (CbCr).

        Returns:
            numpy.ndarray: The inverse quantized channel.

        Notes:
            The inverse quantization process reverses the quantization applied during encoding.
            For luminance (Y) channel, the quantization matrix is fetched from self.header.Q_Y.
            For chrominance (CbCr) channels, the quantization matrix is fetched from self.header.Q_CbCr.
            The inverse quantization is performed by multiplying each element of the quantized
            channel by the corresponding element of the quantization scale factor (qsf).
            If the quantization scale factor (qsf) is obtained from the quality factor (sf) and
            quantization matrix, it's rounded and clipped to the range [1, 255].
            The inverse quantization process is applied on an 8x8 block basis.
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
                    Q_channel[i:i+8, j:j+8] = np.multiply(channel[i:i+8, j:j+8], qsf)
            Q_channel = np.round(Q_channel)
        else:
            Q_channel = channel

        return Q_channel
    
    def apply_iDPCM(self, dpcm):
        """
        Apply Inverse Differential Pulse Code Modulation (iDPCM) to the given channel.

        Args:
            dpcm (numpy.ndarray): The input channel encoded using DPCM.

        Returns:
            numpy.ndarray: The channel after iDPCM decoding.

        Notes:
            Inverse Differential Pulse Code Modulation (iDPCM) is the reverse process of DPCM,
            used to decode the DPCM-encoded signal.
            iDPCM reconstructs the original signal by adding the predicted difference (error)
            to the previously decoded sample.
            This function applies iDPCM on an 8x8 block basis.
            For each 8x8 block in the DPCM-encoded channel, the error values are added to
            reconstruct the original values.
            The reconstruction process starts from the top-left corner of the channel and
            iterates towards the bottom-right corner. The top-left corner is skipped since
            it has no previous values for prediction.
        """
        channel_shape = dpcm.shape
        channel = np.copy(dpcm)

        for r in range(0, channel_shape[0], 8):
            for c in range(0, channel_shape[1], 8):
                if r == 0 and c == 0:
                    continue
                if c == 0:
                    channel[r, 0] = channel[r, 0] + channel[r-8, channel_shape[1]-8]
                else:
                    channel[r, c] = channel[r, c] + channel[r, c-8]

        return channel
