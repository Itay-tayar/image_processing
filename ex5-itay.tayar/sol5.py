import numpy as np
import imageio
from skimage import color
from scipy.ndimage.filters import convolve
import sol5_utils
import tensorflow as tf


GRAY_SCALE_REPRESENT = 1
RGB_REPRESENT = 2
SIZE_REPRESENT_FACTOR = 255
CONV_SIZE = 3, 3



def calculate_64float(im):
    """
    Convert a int 256 image to float 64 image
    :param im:  matrix that represent an image
    :return: matrix that represent the converted image
    """
    im /= SIZE_REPRESENT_FACTOR
    return im


def read_image(filename, representation):
    """
    The function creating a matrix that represent a given image, the output type
    (gray scale or RGB depend on the representation arg)
    :param filename: the name of the image file.
    :param representation: 1 if the output should be gray scale, 2 if RGB.
    :return: the final matrix
    """
    img = imageio.imread(filename)
    im_float = calculate_64float(img.astype(np.float64))

    if len(img.shape) == RGB_REPRESENT:
        return im_float
    else:
        if representation == GRAY_SCALE_REPRESENT:
            im_float = color.rgb2gray(im_float)
    return im_float


def get_rand(im, patch_size):
    """
    get a random height and width for the start of a patch
    :param im: a grayscale image of shape (height, width)
    :param patch_size: shape of the required patch
    :return: height and width for new patch
    """
    h_random = np.random.randint(im.shape[0] - patch_size[0])
    w_random = np.random.randint(im.shape[1] - patch_size[1])
    return h_random, w_random


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Given a set of images, instead of extracting all
    possible image patches and applying a finite set of corruptions, we will generate pairs of image patches
    on the fly, each time picking a random image, applying a random corruption 1 , and extracting a random patch.
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: Python’s generator object which outputs random tuples of the form (source_batch, target_batch)
    """
    image_dict = dict()
    patch_size = (3 * crop_size[0], 3 * crop_size[1])
    while True:
        target_batch, source_batch = [], []

        for i in range(batch_size):
            batch = np.random.choice(filenames)
            if batch not in image_dict:
                image = read_image(batch, GRAY_SCALE_REPRESENT)
                image_dict[batch] = image.reshape(image.shape[0], image.shape[1], 1)
            real_im = image_dict[batch]
            h_random, w_random = get_rand(real_im, patch_size)
            patch = real_im[h_random: h_random + patch_size[0], w_random:w_random + patch_size[1], :]
            corrupt_patch = corruption_func(patch)
            h_ran_patch, w_ran_patch = get_rand(patch, crop_size)
            target_batch.append(patch[h_ran_patch:h_ran_patch + crop_size[0],
                                w_ran_patch:w_ran_patch + crop_size[1], :] - 0.5)
            source_batch.append(corrupt_patch[h_ran_patch: h_ran_patch + crop_size[0],
                                w_ran_patch: w_ran_patch + crop_size[1], :] - 0.5)
        yield (np.array(source_batch), np.array(target_batch))


def resblock(input_tensor, num_channels):
    """
    takes as input a symbolic input tensor and the number of channels for each of its convolutional layers,
    and returns the symbolic output tensor of the layer configuration described in the exercise.
    :param input_tensor: symbolic input tensor
    :param num_channels: number of channels for each of its convolutional layers
    :return: symbolic output tensor
    """
    t = tf.keras.layers.Conv2D(num_channels, CONV_SIZE, padding='same')(input_tensor)
    t = tf.keras.layers.Activation('relu')(t)
    t = tf.keras.layers.Conv2D(num_channels, CONV_SIZE, padding='same')(t)
    t = tf.keras.layers.Add()([input_tensor, t])
    return tf.keras.layers.Activation('relu')(t)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    the function build an untrained model as described in the exercise
    :param height: height of the shape for the input layer
    :param width: width of the shape for the input layer
    :param num_channels: number of channels for each of its convolutional layers
    :param num_res_blocks: number of blocks in the model
    :return: untrained Keras model
    """
    input_tensor = tf.keras.layers.Input(shape=(height, width, 1))
    t = tf.keras.layers.Conv2D(num_channels, CONV_SIZE, padding='same')(input_tensor)
    t = tf.keras.layers.Activation('relu')(t)
    for i in range(num_res_blocks):
        t = resblock(t, num_channels)
    t = tf.keras.layers.Conv2D(1, CONV_SIZE, padding='same')(t)
    t = tf.keras.layers.Add()([input_tensor, t])
    return tf.keras.Model(input_tensor, t)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    divide the images into a training set and validation set,and generate from each set a dataset
    with the given batch size and corruption function, then train the model using fit generator
    :param model: a general neural network model for image restoration.
    :param images: list of file paths pointing to image files. You should assume these paths are complete, and
    should append anything to them.
    :param corruption_func: same as described in section
    :param batch_size: the size of the batch of examples for each iteration of SGD
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    """
    crop_size = model.input_shape[1:3]
    data_generator = load_dataset(images[:round(len(images) * 0.8)], batch_size, corruption_func, crop_size)
    test_generator = load_dataset(images[round(len(images) * 0.8):], batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(beta_2=0.9))
    model.fit_generator(data_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=test_generator, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """

    :param corrupted_image: a grayscale image of shape (height, width)
    :param base_model: a neural network trained to restore small patches
    :return: restored image
    """
    corrupted_image = corrupted_image.reshape(corrupted_image.shape[0], corrupted_image.shape[1], 1) - 0.5
    new_input = tf.keras.layers.Input(shape=corrupted_image.shape)
    new_output = base_model(new_input)
    new_model = tf.keras.Model(new_input, new_output)
    clean_im = new_model.predict(np.array([corrupted_image])) + 0.5
    clean_im = clean_im.reshape(clean_im.shape[1], clean_im.shape[2])
    return clean_im.clip(0, 1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    randomly sample a value of sigma, uniformly distributed between min_sigma and max_sigma, followed by adding to every
    pixel of the input image a zero-mean gaussian random variable with standard deviation equal to sigma.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noised_im = image + np.random.normal(scale=sigma, size=image.shape)
    noised_im = np.around(255*noised_im)/255.0
    return noised_im.clip(0, 1).astype(np.float64)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    create and train a model of denoising images
    :param num_res_blocks: number of blocks in the model
    :param quick_mode: helper argument for less work
    :return: a trained denoising model
    """
    images_for_denoising = sol5_utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 5, 1000
    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    train_model(model, images_for_denoising, lambda im: add_gaussian_noise(im, 0, 0.2), batch_size, steps_per_epoch,
                num_epochs, num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Motion blur by convolving the image with a kernel made of a single line crossing its center, where
    the direction of the motion blur is given by the angle of the line.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π)
    :return: blurred image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    blurred_im = image.copy()
    if len(image.shape) == 3:
        blurred_im[:, :, 0] = convolve(image[:, :, 0], kernel)
    else:
        blurred_im = convolve(image, kernel)
    blurred_im = np.around(255*blurred_im)/255.0
    return blurred_im.clip(0, 1).astype(np.float64)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    amples an angle at uniform from the range [0, π),and choses a kernel size at uniform from the list
    list_of_kernel_sizes, followed by applying the previous function with the given image and the randomly
    sampled parameters.
    :param image:  a grayscale image with values in the [0, 1] range of type float6
    :param list_of_kernel_sizes:   a list of odd integers
    :return: blurred image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    blurred_im = add_motion_blur(image, kernel_size, angle)
    blurred_im = np.around(255*blurred_im)/255.0
    return blurred_im.clip(0, 1).astype(np.float64)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    create and train a model of denblurring images
    :param num_res_blocks: number of blocks in the model
    :param quick_mode: helper argument for less work
    :return: a trained deblurring model
    """
    images_for_deblurring = sol5_utils.images_for_deblurring()
    height, width, num_channels = 16, 16, 32
    model = build_nn_model(height, width, num_channels, num_res_blocks)
    batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 10, 1000
    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    train_model(model, images_for_deblurring, lambda im: random_motion_blur(im, [7]), batch_size, steps_per_epoch,
                num_epochs, num_valid_samples)
    return model


