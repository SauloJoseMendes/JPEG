import numpy as np
from modules.Encoder import Encoder

class Decoder:
    def __init__(self, encoded_img):
        self.R, self.G, self.B = encoded_img.R, encoded_img.G, encoded_img.B
        self.R, self.G, self.B = self.remove_padding(self.R), self.remove_padding(self.G), self.remove_padding(self.B)
        self.RGB = self.join_rgb(self.R, self.G, self.B)

    def join_rgb(self, r, g, b):
        nl, nc = r.shape
        img_rec = np.zeros((nl, nc, 3), dtype=np.uint8)
        img_rec[:, :, 0] = r
        img_rec[:, :, 1] = g
        img_rec[:, :, 2] = b
        return img_rec

    def remove_padding(self, channel):
        channel = self.remove_horizontal_padding(channel)
        channel = self.remove_vertical_padding(channel)
        return channel
    def remove_horizontal_padding(self, channel):
        last_row = channel[-1, :]
        for i in range(channel.shape[0] - 1, -1, -1):
            if not np.array_equal(last_row, channel[i, :]):
                return channel[0:i + 2, :]

    def remove_vertical_padding(self, channel):
        last_column = channel[:, -1]
        for i in range(channel.shape[1] - 1, -1, -1):
            if not np.array_equal(last_column, channel[:, i]):
                return channel[:, 0 : i + 2]
