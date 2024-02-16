import numpy as np
from modules.Encoder import Encoder

class Decoder:
    def __init__(self, encoded_img):
        self.encoded_img = encoded_img
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
        return channel[: self.encoded_img.rows, : self.encoded_img.cols]
