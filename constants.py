import numpy as np
import tensorflow as tf

from Autoencoders import add_gaussian_noise


mnist = tf.keras.datasets.mnist

def load_auto_encoder_mnist_data(noise_sigma=None, with_test=False):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    if with_test:
        if noise_sigma:
            return np.array([add_gaussian_noise(x, noise_sigma)
                             for x in x_train]), x_train, y_train, \
                   np.array([add_gaussian_noise(x, noise_sigma)
                             for x in x_test]), x_test, y_test
        return x_train, x_train, y_train, x_test, x_test, y_test

    images = np.concatenate((x_train, x_test), axis=0)
    noisy_images = images
    if noise_sigma:
        noisy_images = np.array([add_gaussian_noise(image, noise_sigma)
                                 for image in images])
    return noisy_images, images


def load_gan_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mean = 255.0 / 2
    x_train, x_test = (x_train - mean) / mean, (x_test - mean) / mean
    images = np.concatenate((x_train, x_test), axis=0)  # normalized to [-1,1]
    return x_train.reshape(x_train.shape[0], 28, 28, 1)
