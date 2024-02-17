import matplotlib.pyplot as plt

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = plt.imread(image_path)

    def show_img(self, caption="", cmap=None, axis="off"):
        plt.figure()
        plt.imshow(self.image, cmap)
        plt.axis(axis)
        plt.title(caption)
        plt.show(block=True)
        plt.interactive(False)
