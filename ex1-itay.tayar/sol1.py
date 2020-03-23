import scipy as sp
import numpy as np
from imageio import imread
from skimage import color
import matplotlib.pyplot as plt

YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
GRAY_SCALE_REPRESENT = 1
RGB_REPRESENT = 2
FIRST_ROW = 0
SECOND_ROW = 1
THIRD_ROW = 2
SIZE_REPRESENT_FACTOR = 255
RGB_IMAGE_LEN = 3
LAST_INDEX = 255
FIRST_INDEX = 0
HIST_SIZE = np.arange(257)
BEGINNING_OF_SUM = 0
FIRST_ELEMENT = 0
LAST_ELEMENT = 255


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
            im_float = color.rgb2gray(im_float)
    return im_float


def imdisplay(filename, representation):
    """
    display a given image file according to the representation (gray scale or RGB)
    :param filename: the name of the image file.
    :param representation: 1 if the output should be gray scale, 2 if RGB.
    :return:
    """
    plt.imshow(read_image(filename, representation), cmap=plt.cm.gray)
    plt.show()


def new_matrix(imRGB, matrix, row):
    """
    Create new matrix with matrix multiply of the image matrix, and convert matrix.
    :param imRGB: the original image to convert
    :param matrix: convert matrix
    :param row: the row of the convert matrix to use.
    :return: the new matrix
    """

    sum1 = np.add(np.multiply(imRGB[:, :, 0], matrix[row, 0]), np.multiply(imRGB[:, :, 1], matrix[row, 1]))
    sum2 = np.add(sum1, np.multiply(imRGB[:, :, 2],  matrix[row, 2]))

    return sum2


def stack_matrix(im, matrix):
    """
    the function take 3 dimension matrix , convert it to 3 matrix with 1 dimension and different
    values and make it 3 dimension matrix.
    :param im: 3 dims matrix
    :param matrix: matrix to use for converting
    :return: the new 3 dims matrix
    """
    X = new_matrix(im, matrix, FIRST_ROW)
    Y = new_matrix(im, matrix, SECOND_ROW)
    Z = new_matrix(im, matrix, THIRD_ROW)
    return np.stack((X, Y, Z), axis=2)


def rgb2yiq(imRGB):
    """
    Convert a RGB image to YIQ image
    :param imRGB: matrix that represent RGB image
    :return: matrix that represent the converted image
    """
    return stack_matrix(imRGB, YIQ_MATRIX)


def yiq2rgb(imYIQ):
    """
    Convert a YIQ image to RGB image
    :param imYIQ:  matrix that represent YIQ image
    :return:  matrix that represent the converted image
    """
    rgb_matrix = np.linalg.inv(YIQ_MATRIX)
    return stack_matrix(imYIQ, rgb_matrix)


def calculate_256int(im):
    """
    Convert a float 64 image to int 256 image
    :param im: matrix that represent an image
    :return: matrix that represent the converted image
    """
    im *= SIZE_REPRESENT_FACTOR
    return im


def calculate_64float(im):
    """
    Convert a int 256 image to float 64 image
    :param im:  matrix that represent an image
    :return: matrix that represent the converted image
    """
    im /= SIZE_REPRESENT_FACTOR
    return im


def initialize_image(im):
    """
    The function take 64 float image, check it's RGB or grayscale and return the Y matrix or
    grayscale matrix accordingly in 256int representation.
    :param im: 64 float image to convert
    :return: matrix that represent the converted image
    """
    im_orig_yiq = im
    if len(im.shape) == RGB_IMAGE_LEN:
        im_orig_yiq = rgb2yiq(im)
        im_to_work_on = calculate_256int(im_orig_yiq[:, :, 0]).round().astype(np.int64)
    else:
        im_to_work_on = calculate_256int(im).round().astype(np.int64)
    return im_to_work_on, im_orig_yiq


def histogram_equalize(im_orig):
    """
    The function take an image, find it's histogram equalization representation and the equalize image
    :param im_orig: 64 float image to convert
    :return: equalize image, original histogram, equalize histogram
    """
    im_to_work_on, im_orig_yiq = initialize_image(im_orig)
    im_hist = np.histogram(im_to_work_on, HIST_SIZE, (FIRST_INDEX, LAST_INDEX))[FIRST_INDEX]
    cumuli_hist = np.cumsum(im_hist)

    first_nonzero = cumuli_hist[np.flatnonzero(cumuli_hist)[FIRST_INDEX]]

    eq_hist = np.round(((cumuli_hist - first_nonzero)/(cumuli_hist[LAST_INDEX] - first_nonzero)) *
                       SIZE_REPRESENT_FACTOR)
    eq_hist = eq_hist.astype(np.int64)

    im_eq = eq_hist[im_to_work_on]
    final_hist = np.histogram(im_eq, HIST_SIZE, (FIRST_INDEX, LAST_INDEX))[FIRST_INDEX]

    im_eq = im_eq.astype(np.float64)
    im_eq = calculate_64float(im_eq)

    if len(im_orig.shape) == RGB_IMAGE_LEN:
        im_orig_yiq[:, :, FIRST_INDEX] = im_eq
        im_eq = yiq2rgb(im_orig_yiq)

    return im_eq, im_hist, final_hist


def quantize(im_orig, n_quant, n_iter):
    """
    The function does quantization to a given image on n_quants, with n_iterations
    :param im_orig: 64 float image to convert
    :param n_quant: num of quanta to use in the process
    :param n_iter: num of iterations to use in the process
    :return: the quantize image, error array that each element is the error of the i'th iteration.
    """
    im_to_work_on, im_orig_yiq = initialize_image(im_orig)
    z_array, im_hist = initialize_z(im_to_work_on, n_quant)
    error_arr, q_array = [], []

    for i in range(n_iter):
        q_array = set_q(z_array, n_quant, im_hist)
        temp_z_array = set_z(q_array, n_quant)
        if z_array == temp_z_array:
            break
        error = calculate_error(im_hist, temp_z_array, q_array)
        error_arr.append(error)
        z_array = temp_z_array

    table = np.array(build_table(z_array, q_array)).astype(np.int64)
    quantize_im = table[im_to_work_on]
    quantize_im = calculate_64float(quantize_im.astype(np.float64))

    if len(im_orig.shape) == RGB_IMAGE_LEN:
        im_orig_yiq[:, :, FIRST_INDEX] = quantize_im
        quantize_im = yiq2rgb(im_orig_yiq)

    return quantize_im, error_arr


def initialize_z(im, n_quant):
    """
    Initialize an array of bounds that are separate in such a way that
    between each two there going to be the same number of pixels
    :param im: 256 int image
    :param n_quant: number of quanta
    :return: array of the bounds and the histogram of the image
    """
    im_hist = np.histogram(im, HIST_SIZE, (FIRST_INDEX, LAST_INDEX))[FIRST_INDEX]
    cumuli_hist = np.cumsum(im_hist)
    total_pixel = cumuli_hist[-1]
    z_array = [FIRST_ELEMENT] * (n_quant+1)
    z_array[n_quant] = LAST_INDEX

    for i in range(n_quant):
        temp = np.searchsorted(cumuli_hist, (total_pixel/n_quant) * i)
        z_array[i] = temp

    return z_array, im_hist


def set_q(z_array, n_quant, im_hist):
    """
    Create new array of quanta according to the bounds array, num of quants and the histogram
    of the image.
    :param z_array: array of bounds
    :param n_quant: number of quanta
    :param im_hist: histogram of the image
    :return: new quanta array
    """
    q_array = [FIRST_ELEMENT] * n_quant

    for i in range(n_quant):
        sum_up, sum_down = BEGINNING_OF_SUM, BEGINNING_OF_SUM

        for j in range(z_array[i], z_array[i+1]+1):
            sum_up += (j * im_hist[j])
            sum_down += im_hist[j]

        q_array[i] = int(sum_up/sum_down)

    return q_array


def set_z(q_array, n_quant):
    """
    Create new array of bounds according to the quant array and num of quants
    :param q_array: quanta array
    :param n_quant: number of quanta
    :return: new array of bounds
    """
    z_array = [FIRST_ELEMENT] * (n_quant + 1)
    z_array[n_quant] = LAST_ELEMENT
    for i in range(n_quant-1):
        new_z = int((q_array[i]+q_array[i+1])/2)
        z_array[i+1] = new_z

    return z_array


def calculate_error(im_hist, z_array, q_array):
    """
    The function is calculating the error of each iteration.
    :param im_hist: the histogram of the image
    :param z_array: array of bounds
    :param q_array: array of quanta
    :return: the error
    """
    error = BEGINNING_OF_SUM

    for i in range(len(q_array)):
        sum_in = BEGINNING_OF_SUM
        for j in range(z_array[i], z_array[i+1]+1):
            sum_in += ((q_array[i] - j) * (q_array[i] - j)) * im_hist[j]
        error += sum_in

    return error


def build_table(z_array, q_array):
    """
    build new lookup table with q different values
    :param z_array: array of bounds
    :param q_array: array of quanta
    :return: new lookup table
    """
    table = []

    for i in range(len(q_array)):
        size = z_array[i+1] - z_array[i]
        array_to_enter =[q_array[i]] * size
        table.extend(array_to_enter)
    table.extend([q_array[-1]])
    return table


