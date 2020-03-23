from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import scipy
from scipy.signal import convolve2d

from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy import signal


RGB_SHAPE = 3
GRAY_REP = 1
NORM_FACTOR = 255
FILTER = np.array([[1, 1]])
MIN_SIZE = 16


GRAY_SCALE_REPRESENT = 1
RGB_REPRESENT = 2


def calculate_64float(im):
    """
    Convert a int 256 image to float 64 image
    :param im:  matrix that represent an image
    :return: matrix that represent the converted image
    """
    im /= 256
    return im


def read_image(filename, representation):
    """
    The function creating a matrix that represent a given image, the output type
    (gray scale or RGB depend on the representation arg)
    :param filename: the name of the image file.
    :param representation: 1 if the output should be gray scale, 2 if RGB.
    :return: the final matrix
    """
    img = imread(filename)
    im_float = calculate_64float(img.astype(np.float64))

    if len(img.shape) == RGB_REPRESENT:
        return im_float
    else:
        if representation == GRAY_SCALE_REPRESENT:
            im_float = rgb2gray(im_float)
    return im_float


def filter_vector(filter_size):
    """
    Creates the filter vector.
    :param filter_size: The size of the filter vector to create.
    :return: The filter vector that was created.
    """
    filter_vec = FILTER
    while filter_vec.shape[1] < filter_size:
        filter_vec = signal.convolve(filter_vec, FILTER)
    filter_vec = filter_vec / np.sum(filter_vec)
    return filter_vec


def reduce(im, filter_vec):
    """
    Reduce the size of the image by blurring with the filter_vec.
    :param im: The image matrix.
    :param filter_vec: The 1D filter vector.
    :return: The reduced size image.
    """
    fil = convolve2d(filter_vec, np.transpose(filter_vec))
    res = convolve(im, fil, mode='constant')
    return res[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Create gaussian pyramid for the given image by using a filter vector with the given size.
    :param im: The image to create pyramid for.
    :param max_levels: The maximum size of the pyramid.
    :param filter_size: The filter vector size.
    :return: The gaussian pyramid that was created and the filter vector.
    """
    cur_im = im
    pyr = []
    filter_vec = filter_vector(filter_size)
    for i in range(max_levels):
        pyr.append(cur_im)
        if cur_im.shape[0] / 2 < MIN_SIZE or cur_im.shape[1] / 2 < MIN_SIZE:
            break
        cur_im = reduce(cur_im, filter_vec)
    return pyr, filter_vec

# def gauss(n=1):
#     """
#     The function is calculating the gaussian filter with a given size
#     :param n: the size of the gaussian filter
#     :return: gaussian filter of size n.
#     """
#     gauss_val = np.array([[1, 1]]).astype(np.float64)
#     res = np.array([[1, 1]])
#     if n == 1:
#         return np.array([[1]])
#     for i in range(n - 2):
#         res = scipy.signal.convolve(res, gauss_val)
#     return (1 / 2 ** (n - 1)) * res
#
#
# def reduce(filtered_im):
#     """
#     reduce the size of image
#     :param filtered_im:
#     :return:
#     """
#     return filtered_im[1::2, 1::2]
#
#
# def expand(image):
#     s1, s2 = image.shape
#     temp = np.zeros((s1 * 2, s2 * 2))
#     temp[::2, ::2] = image
#     return temp
#
#
# def build_gaussian_pyramid(im, max_levels, filter_size):
#     """
#     The function build a gaussian pyramid with given level using a gaussian filter
#     :param im: image to build from
#     :param max_levels: levels of the pyramid
#     :param filter_size: size of gaussian filter
#     :return: list of each level of the pyramid
#     """
#     pyr = []
#     gauss_filter = gauss(filter_size)
#     filtered_im = im
#     for i in range(max_levels):
#         if filtered_im.shape[0] <= 32 or filtered_im.shape[1] <= 32:
#             break
#         pyr.append(filtered_im)
#         temp = scipy.ndimage.convolve(filtered_im, gauss_filter)
#         temp = scipy.ndimage.convolve(temp, np.transpose(gauss_filter))
#         filtered_im = reduce(temp)
#     return pyr, gauss_filter

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img
