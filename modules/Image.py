import matplotlib.pyplot as plt

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
