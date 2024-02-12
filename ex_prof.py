from modules import plt, clr, np
from modules.Encoder import Encoder
from modules.Image import Image


def encoder(img):
    R, G, B = splitRGB(img)
    return R, G, B


def splitRGB(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    print(R.shape)
    return R, G, B


def decoder(R, G, B):
    imgRec = joinRGB(R, G, B)
    return imgRec


def joinRGB(R, G, B):
    nl, nc = R.shape
    imgRec = np.zeros((nl, nc, 3), dtype=np.uint8)
    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    return imgRec


def showImg(img, caption="", cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis("off")
    plt.title(caption)
    plt.show(block=True)
    plt.interactive(False)


def main():
    fname = "airport.bmp"
    img = plt.imread("imagens/" + fname)
    print(img.shape)
    print(img.dtype)

    # ....
    showImg(img, "Original Image")

    # .... colormap
    cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
    cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
    cm_gray = clr.LinearSegmentedColormap.from_list("gray", [(0, 0, 0), (1, 1, 1)], N=256)

    R, G, B = encoder(img)
    print("-----")
    print(R.shape)
    ##showImg(R, "Red", cm_red)
    I1 = Image("imagens/" + fname)
    R,G,B = Encoder(I1).R,Encoder(I1).G,Encoder(I1).B
    showImg(joinRGB(R,G,B), "padded")
    decoder(R, G, B)


main()
