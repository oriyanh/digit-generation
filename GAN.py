import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, \
    Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Model


LATENT_DIM = 100
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

d_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
g_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='gen_train_accuracy')
g_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
disc_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
loss_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def loss_discriminator_obj(real, fake):
    real_loss = loss_cross_entropy(tf.ones_like(real), real)
    fake_loss = loss_cross_entropy(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def loss_generator_obj(fake):
    return loss_cross_entropy(tf.ones_like(fake), fake)

class GANDiscriminator(Model):

    def __init__(self):
        super(GANDiscriminator, self).__init__()
        self.conv1 = Conv2D(64, 5, activation=tf.nn.leaky_relu, strides=2,
                            padding='SAME', input_shape=(28, 28, 1))
        self.dropout1 = Dropout(0.3)
        self.conv2 = Conv2D(128, 5, activation=tf.nn.leaky_relu, strides=2,
                            padding='SAME')
        self.dropout2 = Dropout(0.3)
        self.flatten = Flatten()
        self.d1 = Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return self.d1(x)

class GANGenerator(Model):

    def __init__(self, latent_dim):
        super(GANGenerator, self).__init__()
        self.d1 = Dense(7 * 7 * 256, input_dim=latent_dim, use_bias=False)
        self.bn1 = BatchNormalization()
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()

        self.resh = Reshape((7, 7, 256))
        self.conv1t = Conv2DTranspose(128, 5, strides=1,
                                      padding='SAME', input_shape=(7, 7, 256), use_bias=False)
        self.bn2 = BatchNormalization()
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.conv2t = Conv2DTranspose(64, 5, strides=2,
                                      padding='SAME', input_shape=(7, 7, 128), use_bias=False)
        self.bn3 = BatchNormalization()
        self.leaky_relu3 = tf.keras.layers.LeakyReLU()
        self.conv3t = Conv2DTranspose(1, 5, strides=2,
                                      activation='tanh',
                                      padding='SAME', input_shape=(14, 14, 64), use_bias=False)

    def call(self, x):
        x = self.d1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.resh(x)
        x = self.conv1t(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv2t(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        return self.conv3t(x)

def get_train_step_gan(batch_size, latent_dim):
    """ Wrapper for training step, needed if running more than one model
    per run
    :return: train step function
    """

    @tf.function
    def train_step(generator, discriminator, im_batch):
        noise = sample_Z(batch_size, latent_dim)
        with tf.GradientTape() as gan_grad_tape:
            with tf.GradientTape() as disc_grad_tape:
                gen_images = generator(noise, training=True)
                preds_real = discriminator(im_batch, training=True)
                preds_fake = discriminator(gen_images, training=True)
                loss_gen = loss_generator_obj(preds_fake)
                loss_disc = loss_discriminator_obj(preds_real, preds_fake)
        disc_grads = disc_grad_tape.gradient(loss_disc,
                                             discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads,
                                           discriminator.trainable_variables))

        gen_grads = gan_grad_tape.gradient(loss_gen, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        d_train_loss(loss_disc)
        g_train_loss(loss_gen)
        g_train_accuracy(tf.ones_like(preds_fake), preds_fake)

    return train_step

def sample_Z(batch_size, latent_dim):
    return tf.random.normal([batch_size, latent_dim])

def train(generator, discriminator, images, latent_dim, num_epochs, batch_size):
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
    sample_noise = sample_Z(16, latent_dim)
    shuffle_seed = 60000
    train_ds = tf.data.Dataset.from_tensor_slices(images) \
        .shuffle(shuffle_seed) \
        .batch(batch_size)

    train_step = get_train_step_gan(batch_size, latent_dim)
    for epoch in range(num_epochs):
        for image_batch in train_ds:
            train_step(generator, discriminator, image_batch)
        print(f'Epoch {epoch + 1} : Disc loss: {d_train_loss.result()}, Gen loss: {g_train_loss.result()}')
        # Reset the metrics for the next epoch
        d_train_loss.reset_states()
        g_train_loss.reset_states()

        generated_images_tensor = generator(sample_noise, training=False)
        fig = plt.figure(figsize=(4, 4))

        for i in range(generated_images_tensor.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images_tensor[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.show()
