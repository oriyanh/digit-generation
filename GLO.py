import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, \
    Reshape, BatchNormalization


l2_loss = tf.keras.losses.MSE
optimizer_model = tf.keras.optimizers.Adam(0.001)
optimizer_latent = tf.keras.optimizers.Adam(0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')

class GLO(Model):

    def __init__(self, latent_dim):
        super(GLO, self).__init__()
        self.d1 = Dense(1024, activation='relu', input_dim=latent_dim)
        self.bn1 = BatchNormalization()

        self.d2 = Dense(7 * 7 * 128, activation='relu')
        self.bn2 = BatchNormalization()

        self.resh = Reshape((7, 7, 128))

        self.conv1t = Conv2DTranspose(64, 4, strides=2, activation='relu',
                                      padding='SAME')
        self.bn3 = BatchNormalization()

        self.conv2t = Conv2DTranspose(1, 4, strides=2, activation='sigmoid',
                                      padding='SAME')

    def call(self, z):
        x = self.d1(z)
        x = self.bn1(x)

        x = self.d2(x)
        x = self.bn2(x)

        x = self.resh(x)

        x = self.conv1t(x)
        x = self.bn3(x)

        return self.conv2t(x)

def project(z):
    return z / np.linalg.norm(z, axis=1)[..., np.newaxis]

def load_mnist_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    images = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)
    return images, labels

def init_dataset(latent_dim):
    images, labels = load_mnist_data()
    z = np.random.normal(size=(images.shape[0], latent_dim))
    latent_vectors = tf.Variable(project(z), trainable=True)
    return images, labels, latent_vectors

def get_train_step(loss_obj):
    # @tf.function
    def train_step(model, latent_vectors, target_images):
        with tf.GradientTape() as tape:
            generated_images = model(latent_vectors)
            loss = tf.reduce_mean(loss_obj(generated_images, target_images))

        gradients = tape.gradient(loss, [latent_vectors, *model.trainable_variables])
        optimizer_latent.apply_gradients(zip(gradients[:1], [latent_vectors]))
        optimizer_model.apply_gradients(zip(gradients[1:], model.trainable_variables))
        # optimizer.apply_gradients((vector_gradients, latent_vectors))
        train_loss(loss)
        # train_accuracy(labels, pred)

    return train_step

def train(model, images, latent_vectors, loss_obj, num_epochs, batch_size):
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
    train_set = latent_vectors.numpy()
    target_set = images[..., tf.newaxis]
    target_indices_set = np.arange(target_set.shape[0])
    train_ds = tf.data.Dataset.from_tensor_slices(
        (target_set, target_indices_set)).shuffle(
        shuffle_seed).batch(batch_size)

    train_step = get_train_step(loss_obj)
    for epoch in range(num_epochs):
        step = 1
        for target_batch, index_batch in train_ds:
            vector_batch = train_set[index_batch.numpy()]
            trainable_vectors = tf.Variable(vector_batch, trainable=True)
            train_step(model, trainable_vectors, target_batch)
            if step % 200 == 0:
                print(
                    f'Epoch {epoch + 1}, {100 * 32 * step / train_set.shape[0]:.2f}%: Train loss: {train_loss.result()}')
            step += 1
            new_latent_vectors = trainable_vectors.numpy()
            train_set[index_batch.numpy()] = project(new_latent_vectors)
        print(f'Epoch {epoch + 1} 100%: Train loss: {train_loss.result()}')
        # Reset the metrics for the next epoch
        train_loss.reset_states()

    return train_set
