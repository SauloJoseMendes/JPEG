import matplotlib.pyplot as plt
import matplotlib.colors as clr

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = plt.imread(image_path)

    @staticmethod
    def show_img(channel, cmap=None, caption="", axis = 'off', subplot = None, first = False, last = False):
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
        colors = [first_color,second_color]
        return clr.LinearSegmentedColormap.from_list(name, colors, N)