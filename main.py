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
    """
    airport.show_img(channel = airport.image, caption = "Img Orig", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.R, cmap = "Reds", caption = "R", subplot = 222)
    Image.show_img(channel = airport_encoded.G, cmap = "Greens", caption = "G", subplot = 223)
    Image.show_img(channel = airport_encoded.B, cmap = "Blues", caption = "B", subplot = 224, last = True)
    """

    # ... ALÍNEA 4 ...
    print("CANAL R ORIGINAL\t" + str(airport.image[:, :, 0].shape) +
          "\nCANAL R COM PADDING\t" + str(airport_encoded.R.shape) +
          "\nCANAL R SEM PADDING\t" + str(airport_decoded.R.shape))
    
    # ... ALÍNEA 5 ...
    """
    Image.show_img(channel = airport_encoded.Y, cmap = "Grays", caption = "Y", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.Cb, cmap = "Blues", caption = "Cb", subplot = 222)
    Image.show_img(channel = airport_encoded.Cr, cmap = "Reds", caption = "Cr", subplot = 223)
    Image.show_img(channel = airport_decoded.RGB, caption = "Depois", subplot = 224, last = True)
    """
    
    print("PIXEL DE COORDENADA [0, 0] ANTES\t" + str(airport.image[0][0]) +
        "\nPIXEL DE COORDENADA [0, 0] DEPOIS\t" + str(airport_decoded.RGB[0][0]))
