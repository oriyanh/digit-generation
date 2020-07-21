import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, \
    Reshape
from tensorflow.keras import Model

LATENT_DIM = 10

train_loss_autoencoder = tf.keras.metrics.Mean(name='train_loss_ae')
optimizer = tf.keras.optimizers.Adam()
loss_obj = tf.keras.losses.MeanSquaredError()


class EncoderModel(Model):

    def __init__(self):
        super(EncoderModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation=tf.nn.leaky_relu, strides=2,
                            padding='SAME')
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, strides=2,
                            padding='SAME')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation=tf.nn.leaky_relu)
        self.d2 = Dense(10, activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class DecoderModel(Model):

    def __init__(self):
        super(DecoderModel, self).__init__()
        self.d1 = Dense(512, activation=tf.nn.leaky_relu)
        self.d2 = Dense(7 * 7 * 64, activation=tf.nn.leaky_relu)
        self.resh = Reshape((7, 7, 64))
        self.conv1t = Conv2DTranspose(64, 3, strides=2,
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
        self.conv2t = Conv2DTranspose(32, 3, strides=2,
                                      activation=tf.nn.leaky_relu,
                                      padding='SAME')
        self.conv3t = Conv2DTranspose(1, 3, strides=1,
                                      activation='sigmoid',
                                      padding='SAME')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.resh(x)
        x = self.conv1t(x)
        x = self.conv2t(x)
        return self.conv3t(x)


def add_gaussian_noise(image, sigma):
    """
    Adds random gaussian noise to image with variance randomly chosen from range [min_sigma, max_sigma]
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the
    gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution
    :return: Noisy image
    """
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.around(noisy_image * 255) / 255
    return noisy_image.clip(0, 1)


def get_train_step_autoencoder():
    """ Wrapper for training step, needed if running more than one model
    per run
    :return: train step function
    """

    @tf.function
    def train_step(encoder, decoder, images, targets):
        with tf.GradientTape() as tape:
            latent_vectors = encoder(images)
            # print(f"Latent vectors: {latent_vectors.shape}")
            pred = decoder(latent_vectors)
            # print(f"Pred: {pred.shape}")
            loss = loss_obj(targets, pred)

        trainable_vars = [*encoder.trainable_variables,
                          *decoder.trainable_variables]

        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        train_loss_autoencoder(loss)

    return train_step


def train_autoencoder(encoder, decoder, images, target_images, num_epochs,
                      batch_size):
    """ Trains a model (subclassing tf.keras.Model) over MNIST data collection

    :param load_data:
    :param use_full_train_set:
    :param Model model: Model to train, whose __call__() function accepts a
        batch of 28x28 greyscale images and returns a 10-class logits
    :param int num_epochs: Number of epochs to train with
    :param int batch_size: Batch size
    :param train_metric: either `train_loss` or `train_accuracy`
    :param test_metric: either `test_loss` or `test_accuracy`
    :param List metric_scaling_factor: ints [train_metric_scale, test_metric_scale] .
        Scales the value outputted by the metric at each measuring point by this value.
    :returns List: [train_metric_values, test_metric_values]
    """

    shuffle_seed = 10000

    train_set = images[..., tf.newaxis]
    target_set = target_images[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_set, target_set)).shuffle(
        shuffle_seed).batch(batch_size)

    train_step = get_train_step_autoencoder()
    for epoch in range(num_epochs):
        for im_batch, target_batch in train_ds:
            train_step(encoder, decoder, im_batch, target_batch)
        print(
            f'Epoch {epoch + 1}, Train loss: {train_loss_autoencoder.result()}')

        # Reset the metrics for the next epoch
        train_loss_autoencoder.reset_states()


def plot_latent_vectors(encoder, dataset, labels, title, filename):
    latent_vectors = encoder(dataset[..., tf.newaxis]).numpy()
    embeddings = np.array(TSNE().fit_transform(latent_vectors))
    for i in range(LATENT_DIM):
        c = embeddings[labels == i]
        plt.scatter(c[..., 0], c[..., 1], s=.5)
    plt.title(title)
    plt.savefig(filename)
    plt.show()
