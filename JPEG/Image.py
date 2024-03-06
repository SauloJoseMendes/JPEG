import matplotlib.pyplot as plt
import matplotlib.colors as clr

class Image:
    """
    This class provides utility functions for working with images using matplotlib.

    Attributes:
        image_path (str): Path to the input image file.
        image (numpy.ndarray): Loaded image data.
    """
    def __init__(self, image_path):
        """
        Initializes an Image object with the provided image file path.

        Parameters:
            image_path (str): Path to the input image file.
        """
        self.image_path = image_path
        self.image = plt.imread(image_path)

    @staticmethod
    def show_img(channel, cmap=None, caption="", axis = 'off', subplot = None, first = False, last = False):
        """
        Displays an image channel with optional colormap and caption.

        Parameters:
            channel (numpy.ndarray): Image channel to display.
            cmap (str or None): Colormap to use. If None, default colormap is used.
            caption (str): Caption for the image.
            axis (str): Display axis option ('on' or 'off'). Default is 'off'.
            subplot (int or None): Subplot number to display the image. If None, a new figure is created.
            first (bool): Indicates whether this is the first image in a subplot grid. Default is False.
            last (bool): Indicates whether this is the last image in a subplot grid. Default is False.
        """
        if first is True or subplot is None:
            plt.figure()
        if subplot is not None:
            plt.subplot(subplot)
        plt.imshow(channel, cmap)
        plt.title(caption)
        plt.axis(axis)
        plt.interactive(False)
        if(last is True or subplot is None):
            plt.show(block=True)

    @staticmethod
    def create_colormap(name, first_color = (0,0,0), second_color = (1,1,1), N=256):
        """
        Creates a custom colormap with two specified colors.

        Parameters:
            name (str): Name for the colormap.
            first_color (tuple): RGB tuple for the first color. Default is black (0, 0, 0).
            second_color (tuple): RGB tuple for the second color. Default is white (1, 1, 1).
            N (int): Number of colors in the colormap. Default is 256.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: Custom colormap.
        """
        colors = [first_color,second_color]
        return clr.LinearSegmentedColormap.from_list(name, colors, N)
