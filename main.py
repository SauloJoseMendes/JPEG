from modules import plt, clr, np, cv2
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

    airport_encoded_8x8 = Encoder(airport, block_size=8)
    airport_encoded_64x64 = Encoder(airport, block_size=64)

    # ... IMAGES DECODED ...
    airport_decoded = Decoder(airport_encoded)
    nature_decoded = Decoder(nature_encoded)
    geometric_decoded = Decoder(geometric_encoded)

    airport_decoded_8x8 = Decoder(airport_encoded_8x8)
    airport_decoded_64x64 = Decoder(airport_encoded_64x64)

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
    """ 
    print("CANAL R ORIGINAL\t" + str(airport.image[:, :, 0].shape) +
          "\nCANAL R COM PADDING\t" + str(airport_encoded.R.shape) +
          "\nCANAL R SEM PADDING\t" + str(airport_decoded.R.shape))
    """

    # ... ALÍNEA 5 ...
    """
    Image.show_img(channel = airport_encoded.Y, cmap = "Grays", caption = "Y", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.Cb, cmap = "Blues", caption = "Cb", subplot = 222)
    Image.show_img(channel = airport_encoded.Cr, cmap = "Reds", caption = "Cr", subplot = 223)
    Image.show_img(channel = airport_decoded.RGB, caption = "Depois", subplot = 224, last = True)
    
    print("PIXEL DE COORDENADA [0, 0] ANTES\t" + str(airport.image[0][0]) +
        "\nPIXEL DE COORDENADA [0, 0] DEPOIS\t" + str(airport_decoded.RGB[0][0]))
    """

    # ... ALÍNEA 6 ...
    """
    # COMPARAR Cb
    # ORIGINAL
    Image.show_img(channel = airport_encoded.Cb, cmap = "Blues", caption = "WITHOUT DOWNSAMPLING", subplot = 221, first = True)
    # CÚBICA
    airport_encoded.downsample_ycbcr(interpolation=cv2.INTER_CUBIC)
    Image.show_img(channel = airport_encoded.Cb_d, cmap = "Blues", caption = "CUBIC", subplot = 222)
    # LANCZOS
    airport_encoded.downsample_ycbcr(interpolation=cv2.INTER_LANCZOS4)
    Image.show_img(channel = airport_encoded.Cb_d, cmap = "Blues", caption = "CUBIC", subplot = 223)
    # LINEAR
    Image.show_img(channel = airport_encoded.Cb_d, cmap = "Blues", caption = "LINEAR", subplot = 224, last = True)
    
    # COMPARAR Cr
    # ORIGINAL
    Image.show_img(channel = airport_encoded.Cr, cmap = "Reds", caption = "WITHOUT DOWNSAMPLING", subplot = 221, first = True)
    # CÚBICA
    airport_encoded.downsample_ycbcr(interpolation=cv2.INTER_CUBIC)
    Image.show_img(channel = airport_encoded.Cr_d, cmap = "Reds", caption = "CUBIC", subplot = 222)
    # LANCZOS
    airport_encoded.downsample_ycbcr(interpolation=cv2.INTER_LANCZOS4)
    Image.show_img(channel = airport_encoded.Cr_d, cmap = "Reds", caption = "LANCZOS", subplot = 223)
    # LINEAR
    Image.show_img(channel = airport_encoded.Cr_d, cmap = "Reds", caption = "LINEAR", subplot = 224, last = True)

    # RECONSTRUIR
    Image.show_img(channel = airport_encoded.Y, cmap = "Greys", caption = "Y Original", subplot = 321, first = True)
    Image.show_img(channel = airport_decoded.Y_up, cmap = "Greys", caption = "Y Upsampled", subplot = 322)
    Image.show_img(channel = airport_encoded.Cb, cmap = "Blues", caption = "Cb Original", subplot = 323)
    Image.show_img(channel = airport_decoded.Cb_up, cmap = "Blues", caption = "Cb Upsampled", subplot = 324)
    Image.show_img(channel = airport_encoded.Cr, cmap = "Reds", caption = "Cr Original", subplot = 325)
    Image.show_img(channel = airport_decoded.Cr_up, cmap = "Reds", caption = "Cr Upsampled", subplot = 326, last = True)
    """

    # ... ALÍNEA 7 ...
    Image.show_img(channel = airport_decoded.RGB, caption = "Full", subplot = 221, first = True)
    Image.show_img(channel = airport_decoded_8x8.RGB, caption = "8x8", subplot = 222)
    Image.show_img(channel = airport_decoded_64x64.RGB, caption = "64x64", subplot = 223, last = True)
