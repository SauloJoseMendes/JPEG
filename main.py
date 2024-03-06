from JPEG import plt, clr, np, cv2, cm
from JPEG.Encoder import Encoder
from JPEG.Image import Image
from JPEG.Header import Header
from JPEG.Decoder import Decoder

if __name__ == '__main__':
    # ... IMAGES ...
    airport = Image("imagens/airport.bmp")
    nature = Image("imagens/nature.bmp")
    geometric = Image("imagens/geometric.bmp")

    # ... IMAGES ENCODED ...
    airport_encoded = Encoder(airport, Header(block_size = None))
    nature_encoded = Encoder(nature)
    geometric_encoded = Encoder(geometric)

    airport_encoded_lanczos = Encoder(airport, Header(interpolation=cv2.INTER_LANCZOS4))
    airport_encoded_cubic = Encoder(airport, Header(interpolation=cv2.INTER_CUBIC))

    airport_4_2_2 = Encoder(airport, Header(downsampling_rate=422))
    airport_4_2_0 = Encoder(airport, Header(downsampling_rate=420))

    airport_encoded_8x8 = Encoder(airport)
    airport_encoded_64x64 = Encoder(airport, Header(block_size=64))

    airport_encoded_quality10 = Encoder(airport, Header(quality_factor=10))
    airport_encoded_quality25 = Encoder(airport, Header(quality_factor=25))
    airport_encoded_quality50 = Encoder(airport, Header(quality_factor=50))
    airport_encoded_quality75 = Encoder(airport, Header(quality_factor=75))
    airport_encoded_quality100 = Encoder(airport, Header(quality_factor=100))

    # ... IMAGES DECODED ...
    airport_decoded = Decoder(airport_encoded)
    nature_decoded = Decoder(nature_encoded)
    geometric_decoded = Decoder(geometric_encoded)

    airport_decoded_8x8 = Decoder(airport_encoded_8x8)
    airport_decoded_64x64 = Decoder(airport_encoded_64x64)

    airport_decoded_quality10 = Decoder(airport_encoded_quality10)
    airport_decoded_quality25 = Decoder(airport_encoded_quality25)
    airport_decoded_quality50 = Decoder(airport_encoded_quality50)
    airport_decoded_quality75 = Decoder(airport_encoded_quality75)
    airport_decoded_quality100 = Decoder(airport_encoded_quality100)
    # ... COLORMAPS ...
    cm_red= Image.create_colormap("cm_red", second_color=(1,0,0))
    cm_green = Image.create_colormap("cm_green", second_color=(0,1,0))
    cm_blue = Image.create_colormap("cm_blue", second_color=(0,0,1))
    cm_grey = Image.create_colormap("cm_grey")

    # ... ALÍNEA 3 ...
    """ 
    airport.show_img(channel = airport.image, caption = "Img Orig", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.R, cmap = cm_red, caption = "R", subplot = 222)
    Image.show_img(channel = airport_encoded.G, cmap = cm_green, caption = "G", subplot = 223)
    Image.show_img(channel = airport_encoded.B, cmap = cm_blue, caption = "B", subplot = 224, last = True)
    """

    # ... ALÍNEA 4 ...
    """ 
    print("CANAL R ORIGINAL\t" + str(airport.image[:, :, 0].shape) +
          "\nCANAL R COM PADDING\t" + str(airport_encoded.R.shape) +
          "\nCANAL R SEM PADDING\t" + str(airport_decoded.R.shape))
    """

    # ... ALÍNEA 5 ...
    """
    Image.show_img(channel = airport_encoded.Y, cmap = cm_grey, caption = "Y", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.Cb, cmap = cm_grey, caption = "Cb", subplot = 222)
    Image.show_img(channel = airport_encoded.Cr, cmap = cm_grey, caption = "Cr", subplot = 223)
    Image.show_img(channel = airport_decoded.RGB, caption = "Depois", subplot = 224, last = True)
    
    print("PIXEL DE COORDENADA [0, 0] ANTES\t" + str(airport.image[0][0]) +
        "\nPIXEL DE COORDENADA [0, 0] DEPOIS\t" + str(airport_decoded.RGB[0][0]))
    """

    # ... ALÍNEA 6 ...
    """
    # COMPARAR Cb
    Image.show_img(channel = airport_encoded.Cb, cmap = cm_blue, caption = "WITHOUT DOWNSAMPLING", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.Cb_d, cmap = cm_blue, caption = "CUBIC", subplot = 222)
    Image.show_img(channel = airport_encoded.Cb_d, cmap = cm_blue, caption = "LANCZOS", subplot = 223)
    Image.show_img(channel = airport_encoded.Cb_d, cmap = cm_blue, caption = "LINEAR", subplot = 224, last = True)
    
    # COMPARAR Cr
    Image.show_img(channel = airport_encoded.Cr, cmap = cm_red, caption = "WITHOUT DOWNSAMPLING", subplot = 221, first = True)
    Image.show_img(channel = airport_encoded.Cr_d, cmap = cm_red, caption = "CUBIC", subplot = 222)
    Image.show_img(channel = airport_encoded.Cr_d, cmap = cm_red, caption = "LANCZOS", subplot = 223)
    Image.show_img(channel = airport_encoded.Cr_d, cmap = cm_red, caption = "LINEAR", subplot = 224, last = True)
    """
    """
    # RECONSTRUIR
    Image.show_img(channel = airport_encoded.Y, cmap = cm_grey, caption = "Y Original", subplot = 321, first = True)
    Image.show_img(channel = airport_decoded.Y_up, cmap = cm_grey, caption = "Y Upsampled", subplot = 322)
    Image.show_img(channel = airport_encoded.Cb, cmap = cm_blue, caption = "Cb Original", subplot = 323)
    Image.show_img(channel = airport_decoded.Cb_up, cmap = cm_blue, caption = "Cb Upsampled", subplot = 324)
    Image.show_img(channel = airport_encoded.Cr, cmap = cm_red, caption = "Cr Original", subplot = 325)
    Image.show_img(channel = airport_decoded.Cr_up, cmap = cm_red, caption = "Cr Upsampled", subplot = 326, last = True)
    """

    # ESCOLHER A MELHOR (LINEAR)
    """ 
    Image.show_img(channel = airport_4_2_2.Y_d, cmap = cm_grey, caption = "Y downsampling 4:2:2", subplot = 231, first = True)
    Image.show_img(channel = airport_4_2_2.Cb_d, cmap = cm_grey, caption = "Cb downsampling 4:2:2", subplot = 232)
    Image.show_img(channel = airport_4_2_2.Cr_d, cmap = cm_grey, caption = "Cr downsampling 4:2:2", subplot = 233)

    Image.show_img(channel = airport_4_2_0.Y_d, cmap = cm_grey, caption = "Y downsampling 4:2:0", subplot = 234)
    Image.show_img(channel = airport_4_2_0.Cb_d, cmap = cm_grey, caption = "Cb downsampling 4:2:0", subplot = 235)
    Image.show_img(channel = airport_4_2_0.Cr_d, cmap = cm_grey, caption = "Cr downsampling 4:2:0", subplot = 236, last = True)
    """
    # ... ALÍNEA 7 ...
    """
    Image.show_img(channel = np.log(np.abs(airport_encoded.Y_DCT) + 0.0001),cmap=cm_grey, caption = "Full - Y_DCT", subplot = 331, first = True)
    Image.show_img(channel = np.log(np.abs(airport_encoded.Cb_DCT) + 0.0001),cmap=cm_grey, caption = "Full - Cb_DCT", subplot = 332)
    Image.show_img(channel = np.log(np.abs(airport_encoded.Cr_DCT) + 0.0001),cmap=cm_grey, caption = "Full - Cr_DCT", subplot = 333)

    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Y_DCT) + 0.0001),cmap=cm_grey, caption = "8x8 - Y_DCT", subplot = 334)
    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Cb_DCT) + 0.0001),cmap=cm_grey, caption = "8x8 - Cb_DCT", subplot = 335)
    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Cr_DCT) + 0.0001),cmap=cm_grey, caption = "8x8 - Cr_DCT", subplot = 336)

    Image.show_img(channel = np.log(np.abs(airport_encoded_64x64.Y_DCT) + 0.0001),cmap=cm_grey, caption = "64x64 - Y_DCT", subplot = 337)
    Image.show_img(channel = np.log(np.abs(airport_encoded_64x64.Cb_DCT) + 0.0001),cmap=cm_grey, caption = "64x64 - Cb_DCT", subplot = 338)
    Image.show_img(channel = np.log(np.abs(airport_encoded_64x64.Cr_DCT) + 0.0001),cmap=cm_grey, caption = "64x64 - Cr_DCT", subplot = 339, last = True)
    """
    # ... ALÍNEA 8 ...
    """
    for img, quality in ((airport_encoded_quality10,10),
    (airport_encoded_quality25,25),
    (airport_encoded_quality50,50),
    (airport_encoded_quality75,75),
    (airport_encoded_quality100,100)):
        Image.show_img(channel = np.log(np.abs(img.Y_Q) + 0.0001),cmap=cm_grey, caption = "Y_Q "+str(quality)+'%', subplot = 131, first = True)
        Image.show_img(channel = np.log(np.abs(img.Cb_Q) + 0.0001),cmap=cm_grey, caption = "Cb_Q "+str(quality)+'%', subplot = 132)
        Image.show_img(channel = np.log(np.abs(img.Cr_Q) + 0.0001),cmap=cm_grey, caption = "Cr_Q "+str(quality)+'%', subplot = 133 , last = True)
    """

    Image.show_img(channel = airport_decoded_quality10.RGB, caption = "10% Depois", subplot = 231, first = True)
    Image.show_img(channel = airport_decoded_quality25.RGB, caption = "25% Depois", subplot = 232)
    Image.show_img(channel = airport_decoded_quality50.RGB, caption = "50% Depois", subplot = 233)
    Image.show_img(channel = airport_decoded_quality75.RGB, caption = "75% Depois", subplot = 234)
    Image.show_img(channel = airport_decoded_quality100.RGB, caption = "100% Depois", subplot = 235, last = True)

    # ... ALÍNEA 9 ...
    """
    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Y_DPCM) + 0.0001),cmap=cm_grey, caption = "Y_DPCM", subplot = 131, first = True)
    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Cb_DPCM) + 0.0001),cmap=cm_grey, caption = "Cb_DPCM", subplot = 132)
    Image.show_img(channel = np.log(np.abs(airport_encoded_8x8.Cr_DPCM) + 0.0001),cmap=cm_grey, caption = "Cr_DPCM", subplot = 133 , last = True)
    
    print(".....Y_Q ENCODED BLOCK 0 (i-1).....")
    print(airport_encoded_8x8.Y_Q[0:8,0:8])
    print(".....Y_Q ENCODED BLOCK 1 (i).....")
    print(airport_encoded_8x8.Y_Q[0:8,8:16])
    print(".....Y_DPCM ENCODED BLOCK 1 (i).....")
    print(airport_encoded_8x8.Y_DPCM[0:8,8:16])
    print(".....Y_Q DECODED BLOCK 1 (i).....")
    print(airport_decoded_8x8.Y_Q[0:8,8:16])
    """

    # ... ALÍNEA 10 ...

