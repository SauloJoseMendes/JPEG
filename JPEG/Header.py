import cv2
import numpy as np
from scipy.fft import dct

class Header:
    """
    Header class containing metadata and operations related to image processing.

    Attributes:
        conversion_matrix (numpy.ndarray): Matrix for converting between RGB and YCbCr color spaces.
        conversion_matrix_inversed (numpy.ndarray): Inversed conversion matrix.
        YCbCr_offset (numpy.ndarray): Offset for YCbCr color space conversion.
        Q_Y (numpy.ndarray): Quantization matrix for luminance (Y) channel.
        Q_CbCr (numpy.ndarray): Quantization matrix for chrominance (CbCr) channels.

    Methods:
        __init__(self, downsampling_rate=422, interpolation=cv2.INTER_LINEAR, block_size=8, quality_factor=75):
            Initialize the Header object with default or provided parameters.
        calculate_Y_diff(self, Y_0, Y_r):
            Calculate absolute difference between two luminance (Y) channels.
        calculate_max_Y_diff(self, Y_0, Y_r):
            Calculate the maximum absolute difference between two luminance (Y) channels.
        calculate_avg_Y_diff(self, Y_0, Y_r):
            Calculate the average absolute difference between two luminance (Y) channels.
        calculate_MSE(self, img_0, img_r):
            Calculate the Mean Squared Error (MSE) between two images.
        calculate_RMSE(self, img_0, img_r):
            Calculate the Root Mean Squared Error (RMSE) between two images.
        calculate_SNR(self, img_0, img_r):
            Calculate the Signal-to-Noise Ratio (SNR) between two images.
        calculate_PSNR(self, img_0, img_r):
            Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
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

    def __init__(self, downsampling_rate=422, interpolation=cv2.INTER_LINEAR, block_size = 8, quality_factor = 75):
        """
        Initialize the Header object with default or provided parameters.

        Args:
            downsampling_rate (int): Downsampling rate.
            interpolation: Interpolation method.
            block_size (int): Size of image blocks for processing.
            quality_factor (int): Quality factor for image compression.
        """
        self.downsampling_rate = downsampling_rate
        self.interpolation = interpolation
        self.block_size = block_size
        self.rows = 0 
        self.columns = 0
        self.quality_factor = quality_factor

    def calculate_Y_diff(self,Y_0, Y_r):
        """
        Calculate absolute difference between two luminance (Y) channels.

        Args:
            Y_0 (numpy.ndarray): Reference luminance (Y) channel.
            Y_r (numpy.ndarray): Reconstructed luminance (Y) channel.

        Returns:
            numpy.ndarray: Absolute difference between Y_0 and Y_r.
        """
        Y_0_float = Y_0.astype(np.float64)
        Y_r_float = Y_r.astype(np.float64)
        return np.abs(Y_0_float-Y_r_float)
    
    def calculate_max_Y_diff(self,Y_0, Y_r):
        """
        Calculate the maximum absolute difference between two luminance (Y) channels.

        Args:
            Y_0 (numpy.ndarray): Reference luminance (Y) channel.
            Y_r (numpy.ndarray): Reconstructed luminance (Y) channel.

        Returns:
            float: Maximum absolute difference between Y_0 and Y_r.
        """
        return np.max(self.calculate_Y_diff(Y_0, Y_r))
    
    def calculate_avg_Y_diff(self,Y_0, Y_r):
        """
        Calculate the average absolute difference between two luminance (Y) channels.

        Args:
            Y_0 (numpy.ndarray): Reference luminance (Y) channel.
            Y_r (numpy.ndarray): Reconstructed luminance (Y) channel.

        Returns:
            float: Average absolute difference between Y_0 and Y_r.
        """
        return np.average(self.calculate_Y_diff(Y_0, Y_r))
    
    def calculate_MSE(self, img_0, img_r):
        """
        Calculate the Mean Squared Error (MSE) between two images.

        Args:
            img_0 (numpy.ndarray): Reference image.
            img_r (numpy.ndarray): Reconstructed image.

        Returns:
            float: Mean Squared Error between img_0 and img_r.
        """
        img_0_float = img_0.astype(np.float64)
        img_r_float = img_r.astype(np.float64)
        return np.sum(np.square(img_0_float-img_r_float))/(self.rows * self.columns)
    
    def calculate_RMSE(self, img_0, img_r):
        """
        Calculate the Root Mean Squared Error (RMSE) between two images.

        Args:
            img_0 (numpy.ndarray): Reference image.
            img_r (numpy.ndarray): Reconstructed image.

        Returns:
            float: Root Mean Squared Error between img_0 and img_r.
        """
        return np.sqrt(self.calculate_MSE(img_0=img_0, img_r=img_r))
    
    def calculate_SNR(self, img_0, img_r):
        """
        Calculate the Signal-to-Noise Ratio (SNR) between two images.

        Args:
            img_0 (numpy.ndarray): Reference image.
            img_r (numpy.ndarray): Reconstructed image.

        Returns:
            float: Signal-to-Noise Ratio between img_0 and img_r.
        """
        img_0_float = img_0.astype(np.float64)
        img_r_float = img_r.astype(np.float64)
        p = np.sum(np.square(img_0_float))/(self.rows * self.columns)
        return 10 * np.log10(p/self.calculate_MSE(img_0=img_0_float, img_r=img_r_float))
    
    def calculate_PSNR(self, img_0, img_r):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Args:
            img_0 (numpy.ndarray): Reference image.
            img_r (numpy.ndarray): Reconstructed image.

        Returns:
            float: Peak Signal-to-Noise Ratio between img_0 and img_r.
        """
        img_0_float = img_0.astype(np.float64)
        img_r_float = img_r.astype(np.float64)
        return 10 * np.log10(np.max(np.square(img_0_float))/self.calculate_MSE(img_0=img_0_float, img_r=img_r_float))

