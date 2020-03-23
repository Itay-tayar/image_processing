import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
from skimage import img_as_bool
import os

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


def gauss(n=1):
    """
    The function is calculating the gaussian filter with a given size
    :param n: the size of the gaussian filter
    :return: gaussian filter of size n.
    """
    gauss_val = np.array([[1, 1]]).astype(np.float64)
    res = np.array([[1, 1]])
    if n == 1:
        return np.array([[1]])
    for i in range(n - 2):
        res = scipy.signal.convolve(res, gauss_val)
    return (1 / 2 ** (n - 1)) * res


def reduce(filtered_im):
    """
    reduce the size of image
    :param filtered_im:
    :return:
    """
    return filtered_im[1::2, 1::2]


def expand(image):
    s1, s2 = image.shape
    temp = np.zeros((s1 * 2, s2 * 2))
    temp[::2, ::2] = image
    return temp


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    The function build a gaussian pyramid with given level using a gaussian filter
    :param im: image to build from
    :param max_levels: levels of the pyramid
    :param filter_size: size of gaussian filter
    :return: list of each level of the pyramid
    """
    pyr = []
    gauss_filter = gauss(filter_size)
    filtered_im = im
    for i in range(max_levels):
        if filtered_im.shape[0] <= 32 or filtered_im.shape[1] <= 32:
            break
        pyr.append(filtered_im)
        temp = scipy.ndimage.convolve(filtered_im, gauss_filter)
        temp = scipy.ndimage.convolve(temp, np.transpose(gauss_filter))
        filtered_im = reduce(temp)
    return pyr, gauss_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    The function build a gaussian pyramid with given level using a laplacian filter
    :param im: image to build from
    :param max_levels: levels of the pyramid
    :param filter_size: size of gaussian filter
    :return: list of each level of the pyramid
    """
    gauss_arr, gauss_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    lap_pyr = []
    for i in range(len(gauss_arr) - 1):
        temp = expand(gauss_arr[i + 1])
        temp1 = scipy.ndimage.convolve(temp, gauss_filter * 2)
        temp2 = scipy.ndimage.convolve(temp1, np.transpose(gauss_filter * 2))
        lap_im = gauss_arr[i] - temp2
        lap_pyr.append(lap_im)
    lap_pyr.append(gauss_arr[len(gauss_arr) - 1])
    return lap_pyr, gauss_filter * 2


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the function reconstruct an image from its Laplacian Pyramid
    :param lpyr:  Laplacian pyramid
    :param filter_vec: the filter ussed to create the pyramid
    :param coeff: list of scalar to multiply each level i of the laplacian pyramid
    :return: reconstruct image
    """
    image = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        expand_im = expand(image)
        temp1 = scipy.ndimage.convolve(expand_im, filter_vec)
        temp2 = scipy.ndimage.convolve(temp1, np.transpose(filter_vec))
        image = np.add(temp2, lpyr[i] * coeff[i])
    return image


def res_x_len(levels):
    """
    Calculate the length of axis x for a image include a few images
    :param levels: number of  images
    :return: length of the axis
    """
    sum = 0
    for i in range(levels):
        sum += (1 / (2 ** i))
    return sum


def render_pyramid(pyr, levels):
    """
    create new image with the images in each level of the pyrmaid
    :param pyr: gaussian or laplacian pyramid
    :param levels: number of images
    :return: the new image
    """
    original_im = pyr[0]
    x, y = original_im.shape
    res = np.zeros((x, (int(y * (res_x_len(levels))))))
    last_col = 0
    for i in range(levels):
        row, col = pyr[i].shape
        res[:row:, last_col:(last_col + col):] = (pyr[i]+1)/2
        last_col += col
    return res


def display_pyramid(pyr, levels):
    """
    The function display the stacked pyramid image
    :param pyr: list of images represent the levels of the pyramid
    :param levels: num of levels in the pyramid
    :return:
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1:  grayscale image to be blended
    :param im2:  grayscale image to be blended
    :param mask: mask containing True and False representing which parts of im1 and im2 should appear
    in the resulting im_blend
    :param max_levels: parameter you should use when generating the Gaussian and Laplacian pyramids
    :param filter_size_im:  the size of the Gaussian filter for the images
    :param filter_size_mask: size of the Gaussian filter for the mask
    :return: blended image
    """
    pyr1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    pyr2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    converted_mask = mask.astype(np.float64)
    gauss_pyr, filter3 = build_gaussian_pyramid(converted_mask, max_levels, filter_size_mask)
    new_pyr = []
    coeff = []
    for i in range(len(gauss_pyr)):
        im = (gauss_pyr[i] * pyr1[i]) + ((1 - gauss_pyr[i]) * pyr2[i])
        new_pyr.append(im)
        coeff.append(1)
    new_im = laplacian_to_image(new_pyr, filter3 * 2, coeff)
    return np.clip(new_im, 0, 1)


def blend_rgb_im(im1, im2, mask, max_levels, filter_size_im):
    """
    the function take 2 rgb images and blend each chanel of the images with the given mask and then stack the chanels
    into one rgb blended image.
    :param mask: mask image with bool type
    :param im1: first rgb image
    :param im2: second rgb image
    :param filter_size_im: the size of the Gaussian filter for the images
    :param max_levels: size of the Gaussian filter for the mask
    :return: new blended rgb image
    """
    fi1 = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_im)
    fi2 = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_im)
    fi3 = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_im)
    return np.stack((fi1,  fi2, fi3), axis=2)


def relpath(filename):
    """
    load an image with name filename with it's relative paths.
    :param filename: name of the image
    :return: path of the image
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    first example of image blending
    :return:
    """
    im1 = read_image(relpath("external/cat1.jpg"), RGB_REPRESENT)
    im2 = read_image(relpath("external/suri1.jpg"), RGB_REPRESENT)
    mask = img_as_bool(read_image(relpath("external/catmask2.jpg"), GRAY_SCALE_REPRESENT))
    final_im = plt_4_pics(im1, im2, mask, 7, 5)
    return im1, im2, mask, final_im


def blending_example2():
    """
    second example of image blending
    :return:
    """
    im1 = read_image(relpath("external/snow.jpg"), RGB_REPRESENT)
    im2 = read_image(relpath("external/door.jpg"), RGB_REPRESENT)
    mask = img_as_bool(read_image(relpath("external/doormask.jpg"), GRAY_SCALE_REPRESENT))
    final_im = plt_4_pics(im1, im2, mask, 3, 7)
    return im1, im2, mask, final_im


def plt_4_pics(im1, im2, mask, levels, filter_size):
    """
    The function plot 4 images: im1, im2, mask and blended image of the all that
    :param mask: mask image with bool type
    :param im1: first rgb image
    :param im2: second rgb image
    :param filter_size_im: the size of the Gaussian filter for the images
    :param max_levels: size of the Gaussian filter for the mask
    :return:
    """
    final_im = blend_rgb_im(im1, im2, mask, levels, filter_size)
    im_list = [im1, im2, mask, final_im]
    name_list = ["im1", "im2", "mask", "blended image"]
    for i in range(len(im_list)):
        plt.subplot(2, 2, i + 1)
        if len(im_list[i].shape) == 2:
            plt.imshow(im_list[i], cmap=plt.cm.gray)
        else:
            plt.imshow(im_list[i])
        plt.title(name_list[i])
    plt.show()
    return final_im
