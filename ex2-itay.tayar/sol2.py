import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.io.wavfile import read, write
import scipy.signal
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


GRAY_SCALE_REPRESENT = 1
RGB_REPRESENT = 2
DER_MAT = [[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec




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


def build_omega_matrix(vector, sign):
    """
    build an omega matrix for which it with be possible to calculate the Fourier basis signal
    :param vector: array of dtype float64 with shape (N,1) representing signal
    :param sign: -1 for dft and 1 for idft
    :return: omega matrix with shape (N,N)
    """
    n = len(vector)
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    omega_matrix = np.exp((sign * 2j * np.pi) / n)
    return np.power(omega_matrix, x * y).astype(np.complex128)


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal: array of dtype float64 with shape (N,1)
    :return:  array of dtype complex128 with  (N,1) representing 1D Fourier basis
    """
    omega_matrix = build_omega_matrix(signal, -1)
    fourier = np.matmul(omega_matrix, signal)
    return fourier


def IDFT(fourier_signal):
    """
    transform a 1D signal in Fourier representation to its discrete representation.
    :param fourier_signal: an array of dtype complex128 with shape (N,1)
    :return:  array of dtype complex128 with  (N,1) representing 1D discrete signal.
    """
    omega_mat_inver = build_omega_matrix(fourier_signal, 1) / len(fourier_signal)
    inver_fourier = np.matmul(omega_mat_inver, fourier_signal)
    return np.real_if_close(inver_fourier)


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: 2D array represent grayscale image of dtype float64
    :return: 2D array of dtype complex128
    """
    fourier_im = DFT(image)
    fourier_im = DFT(np.transpose(fourier_im))
    return np.transpose(fourier_im)


def IDFT2(fourier_image):
    """
    convert a 2D signal in Fourier representation to its discrete representation
    :param fourier_image:  2D array of dtype complex128
    :return: 2D array represent grayscale image of dtype float64
    """
    reg_im = IDFT(fourier_image)
    reg_im = IDFT(np.transpose(reg_im))
    return np.transpose(reg_im)


def change_rate(filename, ratio):
    """
    Change the sample rate of the the audio file by multiplying it with the given ratio
    and then save the new info to new audio file.
    :param filename: a string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    :return:
    """
    rate, data = read(filename)  # read the audio file
    new_rate = rate * ratio
    write("change_rate.wav", int(new_rate), data)


def change_samples(filename, ratio):
    """
    Change the samples of the the audio file by resizing them with resize function and then save
    the new info to new audio file.
    :param filename:  a string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    :return: 1D ndarray of dtype float64 representing the new sample points
    """
    rate, data = read(filename)  # read the audio file
    new_data = resize(data, ratio).astype(np.float64)
    write("change_samples.wav", rate, new_data)
    return new_data


def resize(data, ratio):
    """
    change the number of samples by the given ratio
    :param data: a 1D ndarray of dtype float64 or complex128
    :param ratio: positive float64 representing the duration change.
    :return: a 1D ndarray of the dtype of data representing the new sample points
    """
    transform_data = DFT(data)
    shift_data = np.fft.fftshift(transform_data)

    index_of_zero_freq = np.where(transform_data[0] == shift_data)[0][0]
    new_data_len = int((len(data) / ratio))
    difference = (len(data) - new_data_len)

    if difference > 0:  # resize to shorter array
        index_to_start = index_of_zero_freq - int(new_data_len/2)
        shift_data = shift_data[index_to_start: index_to_start + new_data_len]

    elif difference < 0:  # resize to larger array
        shift_data = np.pad(shift_data, ((-difference//2) + ((-difference) % 2), (-difference//2)),
                            'constant', constant_values=0)

    new_data = np.fft.ifftshift(shift_data)
    return IDFT(new_data).astype(data.dtype)


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return:
    """
    spec_data = stft(data)
    new_data = []
    for row in range(len(spec_data)):
        new_data.append(resize(spec_data[row], ratio))
    return istft(np.array(new_data)).astype(np.float64)


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data:  1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return:
    """
    spec_data = stft(data)
    new_data = phase_vocoder(spec_data, ratio)
    return istft(new_data).astype(np.int16)


def conv_der(im):
    """
    computes the magnitude of image derivatives using convolution
    :param im:  2D array represent grayscale image of dtype float64
    :return: magnitude of image derivatives
    """
    x_conv = scipy.signal.convolve2d(im, DER_MAT, 'same')
    y_conv = scipy.signal.convolve2d(im, np.transpose(DER_MAT), 'same')
    return np.sqrt(np.abs(x_conv) ** 2 + np.abs(y_conv) ** 2)


def fourier_der(im):
    """
    computes the magnitude of image derivatives using Fourier transform
    :param im: 2D array represent grayscale image of dtype float64
    :return: magnitude of image derivatives
    """
    n, m = im.shape
    transform_im = np.fft.fftshift(DFT2(im))
    dx = np.zeros((n, m)).astype(np.complex128)
    dy = np.zeros((n, m)).astype(np.complex128)

    for x in range(n):
        dx[x, :] = transform_im[x, :] * (x - n / 2) * 2j * np.pi / n

    for y in range(m):
        dy[:, y] = transform_im[:, y] * (y - m / 2)* 2j * np.pi / m

    non_shift_x = IDFT2(np.fft.ifftshift(dx))
    non_shift_y = IDFT2(np.fft.ifftshift(dy))
    derivative_im = np.sqrt(np.abs(non_shift_x) ** 2 + np.abs(non_shift_y) ** 2)
    return derivative_im

