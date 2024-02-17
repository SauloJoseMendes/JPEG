import numpy as np
from modules.Encoder import Encoder

class Decoder:
    def __init__(self, encoded_img):
        self.encoded_img = encoded_img
        self.R, self.G, self.B = self.convert_to_rgb(encoded_img.Y,encoded_img.Cb,encoded_img.Cr)
        self.R, self.G, self.B = self.remove_padding(self.R), self.remove_padding(self.G), self.remove_padding(self.B)
        self.RGB = self.join_rgb(self.R, self.G, self.B)

    def convert_to_rgb(self, Y, Cb, Cr):
        T = np.array([[0.299, 0.587, 0.144],
                      [-0.168736, -0.331264, 0.5],
                      [0.5, -0.418688, -0.081312]])

        Ti = np.linalg.inv(T)

        ycbcr = np.stack((Y, Cb, Cr))

        rgb = np.dot(Ti, ycbcr.reshape(3, -1)) - np.array([[0], [128], [128]])
        rgb = np.round(rgb)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.reshape(ycbcr.shape)

        return rgb.astype(np.uint8)

    def join_rgb(self, r, g, b):
        nl, nc = r.shape
        img_rec = np.zeros((nl, nc, 3), dtype=np.uint8)
        img_rec[:, :, 0] = r
        img_rec[:, :, 1] = g
        img_rec[:, :, 2] = b
        return img_rec

    def remove_padding(self, channel):
        return channel[: self.encoded_img.rows, : self.encoded_img.cols]
