import numpy as np
class Encoder:
    def __init__(self, img):
        self.img = img
        self.R, self.G, self.B = self.split_rgb(img)
        self.rows, self.cols = self.R.shape
        self.R, self.G, self.B = self.apply_padding(self.R), self.apply_padding(self.G), self.apply_padding(self.B)

    def split_rgb(self, img):
        r = img.image[:, :, 0]
        g = img.image[:, :, 1]
        b = img.image[:, :, 2]
        return r, g, b
    def apply_padding(self, channel):
        channel = self.apply_vertical_padding(channel)
        channel = self.apply_horizontal_padding(channel)
        return channel

    def apply_vertical_padding(self, channel):
        # ........ vertical padding (add rows) ..........
        line_n = channel.shape[0]
        vertical_padding = 32 - (line_n % 32)
        vertical_array = np.repeat(channel[-1:, :], vertical_padding, 0)
        channel = np.vstack((channel, vertical_array))
        return channel

    def apply_horizontal_padding(self, channel):
        # ........ horizontal padding (add colums) ..........
        column_n = channel.shape[1]
        horizontal_padding = 32 - (column_n % 32)
        horizontal_array = np.repeat(channel[:, -1:], horizontal_padding, 1)
        channel = np.hstack((channel, horizontal_array))
        return channel
