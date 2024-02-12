from modules import plt, clr, np
from modules.Encoder import Encoder
from modules.Image import Image
from modules.Decoder import Decoder

if __name__ == '__main__':
    # ... IMAGES ...
    airport = Image("imagens/airport.bmp")
    nature = Image("imagens/nature.bmp")
    geometric = Image("imagens/geometric.bmp")
    # ... IMAGES ENCODED ...
    airport_encoded = Encoder(airport)
    nature_encoded = Encoder(nature)
    geometric_encoded = Encoder(geometric)
    # ... IMAGES DECODED ...
    airport_decoded = Decoder(airport_encoded)
    nature_decoded = Decoder(nature_encoded)
    geometric_decoded = Decoder(geometric_encoded)
    # ... COLORMAPS ...
    cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
    cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
    cm_gray = clr.LinearSegmentedColormap.from_list("gray", [(0, 0, 0), (1, 1, 1)], N=256)

    # ... ALÍNEA 3 ...
    """"
    airport.show_img("airport.bmp")
    airport.show_img("red colormap", cm_red)
    airport.show_img("green colormap", cm_green)
    airport.show_img("blue colormap", cm_blue)
    airport.show_img("gray colormap", cm_gray)
    """
    # ... ALÍNEA 4 ...
    print("CANAL R ORIGINAL\t" + str(airport.image[:, :, 0].shape) +
          "\nCANAL R COM PADDING\t" + str(airport_encoded.R.shape) +
          "\nCANAL R SEM PADDING\t" + str(airport_decoded.R.shape))