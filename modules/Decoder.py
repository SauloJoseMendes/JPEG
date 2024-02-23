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
        self.Y_d, self.Cb_d, self.Cr_d = self.calculate_idct(self.encoded_img.Y_DCT), self.calculate_idct(self.encoded_img.Cb_DCT), self.calculate_idct(self.encoded_img.Cr_DCT)
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
        return channel[: self.encoded_img.rows, : self.encoded_img.cols]
    
    def upsample_ycbcr(self):
        """
        Upsamples the chroma blue (Cb) and chroma red (Cr) components to match the dimensions of the luma (Y) component.

        Returns:
            tuple: Upsampled luma (Y), chroma blue (Cb), and chroma red (Cr) components.
        """
        # Get the dimensions of the original Y channel
        original_height, original_width = self.Y_d.shape
        interpolation = self.encoded_img.interpolation
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
        if self.encoded_img.block_size is None:
            return idct(idct(channel_dct, norm='ortho').T, norm='ortho').T
        
        channel_shape = channel_dct.shape
        channel = np.zeros(channel_shape)

        for i in range(0, channel_shape[0], self.encoded_img.block_size):
            for j in range(0, channel_shape[1], self.encoded_img.block_size):
                channel[i:i+self.encoded_img.block_size, j:j+self.encoded_img.block_size] = idct(idct(channel_dct[i:i+self.encoded_img.block_size, j:j+self.encoded_img.block_size], norm='ortho').T, norm='ortho').T
        
        return channel
