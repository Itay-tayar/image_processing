from sol2 import *
import random
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from scipy.io.wavfile import read
from skimage.color import rgb2gray


def plt1(im, str):
    plt.figure()
    plt.plot(range(len(im)), im)
    plt.title(str)
    plt.show()


def plt2(im, str):
    im = np.real(im)
    plt.figure()
    if len(im.shape) == 2:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.axis('off')
    plt.title(str)
    plt.show()


def FFT1_test(signal):
    plt1(signal, "orignal")
    F = DFT(signal)
    plt1(np.real(F), "my F")
    F_an = np.fft.fft(signal)
    plt1(F_an, "comp F")
    print(np.isclose(F, F_an, 0.01).all())

    f = IDFT(F)
    plt1(np.real(f), "my f")
    f_an = np.fft.ifft(F)
    plt1(f_an, "comp f")
    print("\n\n\n\nmy f:\n", f)
    print("\n\n\n\ncomp f:\n", f_an)
    # print("\n\n\n\ndiff f:\n", np.diff(f,f_an))
    print(np.isclose(f, f_an, 0.01).all())
    print(np.isclose(signal, f, 0.01).all())


"""def DFT2_test(image):
    F_im = DFT2(image)
    plt2(F_im, "my F")
    F_an = np.fft.fft2(image)
    plt2(F_an, "comp F")"""



def main():
    wav_sr, wav_data = read("C:\\Users\\Itay\\PycharmProjects\\ex2-itay.tayar\\aria_4kHz.wav")
    simple = [random.randrange(10) for _ in range(1000)]
    im = read_image("C:\\Users\\Itay\\PycharmProjects\\ex2-itay.tayar\\monkey.jpg", 1)

    FFT1_test(wav_data)
    #FFT1_test(simple)

    #DFT2_test(im)




if __name__ == "__main__":
    main()
