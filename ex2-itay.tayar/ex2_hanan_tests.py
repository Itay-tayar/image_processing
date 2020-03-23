import unittest
import numpy as np
import sol2
import matplotlib.pyplot as plt
from imageio import imread
import skimage.color as ski
import scipy.io.wavfile as wv



IMGS = ["jerusalem.jpeg",
        "low_contrast.jpeg",
        "monkey.jpg"]

WAV = "aria_4kHz.wav"
WAV_CHANGED_FILENAME = "change_rate.wav"

ITERATIONS_NUM = 10
SIGNAL_SAMPLE_NUM = 49
allow_print_metadata = True
allow_print_content = False
allow_plot = False

def print_ndarray(ndarray, name):
    if not allow_print_metadata: return
    print(name, ": ", ndarray.shape)
    if not allow_print_content: return
    print(ndarray)
    if not allow_plot: return
    plt.figure()
    plt.title(name)
    plt.plot(ndarray)
    plt.show()

def get_random_signal(sample_num):
    return np.random.rand(sample_num,1)

def get_sin_signal(sample_num):
    return np.sin(np.arange(sample_num)/2)

def get_random_freq_function(sample_num):
    real = get_random_signal(sample_num)
    imag = get_random_signal(sample_num)
    return real + 1j*imag

def get_random_freq_matrix():
    real = np.random.rand(100,100)
    imag = np.random.rand(100,100)
    return real + 1j * imag

class MyTestCase(unittest.TestCase):

    def test_DTF(self):
        signal = get_random_signal(SIGNAL_SAMPLE_NUM)
        print_ndarray(signal, "signal")
        ck_dft = sol2.DFT(signal)
        print_ndarray(ck_dft, "my dft")
        ex_dft = np.fft.fft(signal.flatten())
        print_ndarray(ex_dft, "np dft")
        print_ndarray(ck_dft.flatten(), "flatten")
        assert np.isclose(ck_dft.flatten(), ex_dft).all()

    def test_IDFT(self):
        signal = get_random_freq_function(SIGNAL_SAMPLE_NUM)
        ck_dft = sol2.IDFT(signal)
        print_ndarray(ck_dft, "my idft")
        ex_dft = np.fft.ifft(signal.flatten())
        print_ndarray(ex_dft, "np idft")
        assert np.isclose(ck_dft.flatten(), ex_dft).all()

    def test_DFT_independent(self):
        for i in range(ITERATIONS_NUM):
            print("--- iteration ",i+1," ---")
            signal = get_random_signal(SIGNAL_SAMPLE_NUM)
            dft = sol2.DFT(signal)
            assert (dft.shape == dft.shape)
            print("DFT shape OK")
            idft = sol2.IDFT(dft)
            assert (idft.shape == dft.shape)
            print("iDFT shape OK")
            assert (np.isclose(idft,signal).all())
            print("dft>idft values")
            assert (np.isreal(idft).all())
            print("dft>idft is real")

    def test_DFT2(self):
        image = read_image(IMGS[0], representation="gray")
        ck_dft = sol2.DFT2(image)
        print_ndarray(ck_dft, "my dft")
        ex_dft = np.fft.fft2(image)
        print_ndarray(ex_dft, "np dft")
        assert np.isclose(ck_dft, ex_dft).all()

    def test_IDFT2(self):
        freq_mat = get_random_freq_matrix()
        ck_idft = sol2.IDFT2(freq_mat)
        ex_idft = np.fft.ifft2(freq_mat)
        assert np.isclose(ck_idft, ex_idft).all()

    def test_DFT2_independent(self):
        image = read_image(IMGS[0], representation="gray")
        dft = sol2.DFT2(image)
        assert (dft.shape == dft.shape)
        print("DFT shape OK")
        idft = sol2.IDFT2(dft)
        assert (idft.shape == dft.shape)
        print("iDFT shape OK")
        assert (np.isclose(idft, image).all())
        print("dft>idft values")
        assert (np.isreal(idft).all())
        print("dft>idft is real")

    def test_change_rate(self):
        ratio = 1.25
        sol2.change_rate(WAV, ratio)
        old_rate = wv.read(WAV)[0]
        a = wv.read(WAV_CHANGED_FILENAME)
        assert a[0] == old_rate*ratio

    def test_conv_der(self):
        return ## fix
        image = read_image(IMGS[0], representation="gray")
        der = sol2.conv_der(image)
        # der_f = sol2.fourier_der(image)

        grad = np.gradient(image)
        dy = grad[0]
        dx = grad[1]
        ex_der = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
        pass

    def test_resize(self):
        ##
        rate0, data0 = wv.read(WAV)

        ## ---- NOTICE ----
        ## using only part of aria.wav for faster computations,
        ## output wav shouldn't sound right (and wont be saved
        # anyway)
        data_even = data0[: 500]
        data_odd = data0[:-9001]

        ## ---- WARNING ----
        ## you can edit ratios with ratios smaller than 1, notice
        ## that idft can be slow due to new size (too big) of resized
        ## data.

        ratios = [1.5, 1.7, 1, (10 / 11), (10/13)]
        datas = [data_even, data_odd]

        for data in datas:
            print()
            if(data.shape[0] % 2 == 0):
                print("---- EVEN SAMPLE NUM")
            else:
                print("---- ODD SAMPLE NUM")

            for ratio in ratios:
                print()
                print("Changing sample with: RATIO = ", ratio)

                new_data = sol2.resize(data, ratio)

                print("original data: ", data.shape, ", ",
                      data.dtype)
                print("resized data: ", new_data.shape, ", ",
                      new_data.dtype)
                new_sample_num = int(data.shape[0]/ratio)
                print("expected: ", new_sample_num)

                assert new_sample_num == new_data.shape[0]
                assert new_data.dtype == data.dtype

                print("-- OK")

    # def test_spectogram_resize(self):
    #     rate0, data0 = wv.read(WAV)
    #
    #     ratio = 2
    #     new_data = sol2.resize_spectogram(data0, ratio)
    #
    #     print("original data: ", data0.shape, ", ",
    #           data0.dtype)
    #     print("resized data: ", new_data.shape, ", ",
    #           new_data.dtype)
    #     new_sample_num = int(data0.shape[0] / ratio)
    #
    #     print("expected: ", new_sample_num)
    #     assert new_sample_num == new_data.shape[0]


    def test_vecoder(self):
        rate0, data0 = wv.read(WAV)

        ratio = 2
        new_data = sol2.resize_spectogram(data0, ratio)

        print("old audio len =", (1/rate0) * data0.shape[0])
        print("new audio len =", (1/rate0) * new_data.shape[0])
        wv.write(WAV_CHANGED_FILENAME, rate0, new_data)

if __name__ == '__main__':
    unittest.main()



def normalize(im):
    """
    returns a normalized representation of the given image
    where float64 type is restricted to range [0,1]
    :param im: image to normalize
    :return: normalized image in float64 type
    """
    if im.dtype != np.float64: # not normalize
        im = im.astype(np.float64)
        im /= 255
    return im




def read_image(filename, representation):
    """
    reads a given image file as a numpy array in a given
    representation.
    :param filename: image to read
    :param representation: code indicating color, 1=gray, 2=rgb
    :return: numpy array
    """
    im = imread(filename)
    if representation == "gray": # grayscale
        im = ski.rgb2gray(im)

    im = normalize(im)

    return im[0:200,0:200]
