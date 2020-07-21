import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### Metrics, operators and constants ###

from Autoencoders import EncoderModel, DecoderModel, train_autoencoder, \
    plot_latent_vectors
from GAN import GANDiscriminator, GANGenerator
import GLO, GAN
from constants import load_auto_encoder_mnist_data, load_gan_mnist_data


LATENT_DIM = 74
BATCH_SIZE = 32
SIGMAS = [0.05, 0.25, 0.5, 0.7, 1]
# SIGMAS = [0.25]
NUM_EPOCHS = 3

### Answer functions ###

def q1_autoencoder():
    print("3 epochs")

    encoder = EncoderModel()
    decoder = DecoderModel()
    train_noised, train_clean, train_labels, test_noised, test_clean, \
    test_labels = load_auto_encoder_mnist_data(with_test=True)
    train_autoencoder(encoder, decoder, train_noised, train_clean, NUM_EPOCHS,
                      BATCH_SIZE)
    # print("Plotting")
    plot_latent_vectors(encoder, test_noised, test_labels,
                        'Reconstruction AE embedded in 2D space',
                        'q1_plot.png')

def q2_autoencoder():
    for sigma in SIGMAS:
        encoder = EncoderModel()
        decoder = DecoderModel()
        train_noised, train_clean, train_labels, \
        test_noised, test_clean, test_labels = \
            load_auto_encoder_mnist_data(sigma, with_test=True)
        train_autoencoder(encoder, decoder, train_noised, train_clean,
                          3, BATCH_SIZE)
        plot_latent_vectors(encoder, test_noised, test_labels,
                            f'Denoising AE embedded in 2D space\nsigma = {sigma}',
                            f'q2_plot_sigma_{sigma}.png')

        plt.figure()
        for i in range(3):
            plt.subplot(3, 3, (3 * i) + 1)
            plt.imshow(test_clean[i], cmap='gray')
            plt.subplot(3, 3, (3 * i) + 2)
            plt.imshow(test_noised[i], cmap='gray')
            plt.subplot(3, 3, (3 * i) + 3)
            out_im = np.reshape(decoder(encoder(
                test_noised[i][np.newaxis, :, :, np.newaxis])).numpy(),
                                (28, 28))
            plt.imshow(out_im, cmap='gray')
        plt.show()

def q3_gan():
    print("v100 - 50 epochs")
    discriminator = GANDiscriminator()
    generator = GAN.GANGenerator(GAN.LATENT_DIM)
    images = load_gan_mnist_data()
    GAN.train(generator, discriminator, images.astype('float32'), GAN.LATENT_DIM, GAN.NUM_EPOCHS,
              GAN.BATCH_SIZE)
    generator_normalized = lambda x: generator(normalize_output(x), training=False)
    q3_interpolation(generator_normalized, GAN.LATENT_DIM)
    z = q3_novel_sample(generator_normalized, GAN.LATENT_DIM)
    q3_linear_sampling(generator_normalized, z[0], z[1], "GAN_linear_interpolation_1.png")
    return generator_normalized

def q4_glo():
    LATENT_DIM = 74
    NUM_EPOCHS = 15
    model = GLO.GLO(LATENT_DIM)
    l2_loss = tf.keras.losses.MSE
    images, labels, vectors = GLO.init_dataset(LATENT_DIM)
    trained_vectors = GLO.train(model, images, vectors, l2_loss, NUM_EPOCHS, BATCH_SIZE)
    q4_novel_sample(model, LATENT_DIM, 5)
    q4_uniform_sample(model, images, labels, trained_vectors)
    q4_interpolation_same_digit(model, images, labels, trained_vectors)
    q4_interpolation_different_digit(model, images, labels, trained_vectors)

def normalize_output(generated_images):
    """ Normalize from [-1,1] to [0, 255] """
    return (generated_images * 127.5) + 127.5

def novel_sample(model, latent_vectors, title, fname):
    generated_images = model(latent_vectors).numpy().reshape(16, 28, 28)
    plt.figure()
    plt.axis('off')
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.imshow(generated_images[i - 1], cmap="gray")
    plt.suptitle(title)
    plt.savefig(fname)
    plt.show()

def q3_novel_sample(generator, latent_dim):
    z = GAN.sample_Z(16, latent_dim)
    title = f"GAN 16 Novel Samples"
    print("Z vectors:")
    print(title)
    fname = f"GAN_16_novel_samples_0{np.random.randint(1, 100)}.png"
    novel_sample(generator, z, title, fname)

    return z.numpy()

def q3_interpolation(generator, z):
    z_perm = tf.random.shuffle(z.copy())
    z_interpolated_same_digit = z + z
    z_interpolated_different_digit1 = z + z_perm
    z_interpolated_different_digit2 = z_perm + z
    generated_im = generator(z).numpy().reshape(6, 28, 28)
    generated_im_perm = generator(z_perm).numpy().reshape(6, 28, 28)
    generated_im_same_d = generator(z_interpolated_same_digit).numpy().reshape(6, 28, 28)
    generated_im_diff_d1 = generator(z_interpolated_different_digit1).numpy().reshape(6, 28, 28)
    generated_im_diff_d2 = generator(z_interpolated_different_digit2).numpy().reshape(6, 28, 28)

    title_template = "Interpolating {} - Addition (Left + Middle = Right)"

    def plot(first, second, res, title, fname):
        plt.figure()
        plt.axis('off')
        for i in range(0, 4):
            j = i * 2 + 1
            plt.subplot(4, 3, j)
            plt.imshow(first[i], cmap="gray")
            plt.subplot(4, 3, j + 1)
            plt.imshow(second[i], cmap="gray")
            plt.subplot(4, 3, j + 2)
            plt.imshow(res[i], cmap="gray")
        plt.suptitle(title)
        plt.savefig(fname)
        plt.show()

    plot(generated_im, generated_im, generated_im_same_d,
         title_template.format("same digits"),
         "GAN_interpolated_same_digits.png")
    plot(generated_im, generated_im_perm, generated_im_diff_d1,
         title_template.format("different digits"),
         "GAN_interpolated_same_digits_1.png")
    plot(generated_im_perm, generated_im, generated_im_diff_d2,
         title_template.format("different digits"),
         "GAN_interpolated_same_digits_2.png")

def q3_linear_sampling(generator, z1, z2, fname):
    plt.figure()
    alpha = np.arange(0., 1.05, 0.052)
    interpolations = np.array([a * z1 + (1 - a) * z2 for a in alpha])
    gen_images = generator(interpolations).numpy().reshape(interpolations.shape[0], 28, 28)
    for i in range(20):
        plt.subplot(2, 10, i + 1)
        plt.imshow(gen_images[i], cmap="gray")
        plt.axis('off')
    plt.savefig(fname)
    plt.show()

def q4_novel_sample(glo, latent_dim, n):
    for i in range(n):
        z = np.random.normal(size=(16, latent_dim))
        z_norm = GLO.project(z)
        title = f"GLO 16 Novel Samples #{i + 1}"
        fname = f"GLO_16_novel_samples_{i + 1}.png"
        novel_sample(glo, z_norm, title, fname)

def q4_uniform_sample(model, images, labels, vectors):
    plt.figure()
    plt.axis('off')
    for n in range(5):
        indices = np.argwhere(labels == n)[:4, 0]
        test_images, test_vectors = images[indices], vectors[indices]
        generated_images = model(test_vectors).numpy()
        generated_images = np.reshape(generated_images, test_images.shape)
        for j in range(4):
            plt.subplot(5, 8, n * 8 + 1 + j * 2)
            plt.imshow(test_images[j], cmap="gray")
            plt.subplot(5, 8, n * 8 + 1 + j * 2 + 1)
            plt.imshow(generated_images[j], cmap="gray")
        plt.suptitle("4 Samples x 5 classes, original vs. generated (w.r.t L2 loss)")
    plt.savefig("GLO_40_samples_uniform.png")
    plt.show()

def q4_interpolation_same_digit(model, images, labels, vectors):
    indices_three = np.argwhere(labels == 3)[:2, 0]
    indices_four = np.argwhere(labels == 4)[:2, 0]
    sample_three_images, sample_three_vectors = images[indices_three], vectors[indices_three]
    sample_four_images, sample_four_vectors = images[indices_four], vectors[indices_four]
    plt.figure()
    plt.axis('off')
    plt.subplot(2, 3, 1)
    plt.imshow(sample_three_images[0], cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(sample_three_images[1], cmap="gray")
    plt.subplot(2, 3, 3)
    interpolation_vec = sample_three_vectors[0] + sample_three_vectors[1]
    interpolated_img = model(interpolation_vec[np.newaxis, ...]).numpy().reshape(28, 28)
    plt.imshow(interpolated_img, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.imshow(sample_four_images[0], cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(sample_four_images[1], cmap="gray")
    plt.subplot(2, 3, 6)
    interpolation_vec = sample_four_vectors[0] + sample_four_vectors[1]
    interpolated_img = model(interpolation_vec[np.newaxis, ...]).numpy().reshape(28, 28)
    plt.imshow(interpolated_img, cmap="gray")
    plt.suptitle("Interpolating same digit - Addition (Left + Middle = Right)")
    plt.savefig("GLO_interpolated_same_digits.png")
    plt.show()

def q4_interpolation_different_digit(model, images, labels, vectors):
    indices_three = np.argwhere(labels == 3)[:2, 0]
    indices_four = np.argwhere(labels == 4)[:2, 0]
    sample_three_images, sample_three_vectors = images[indices_three], vectors[indices_three]
    sample_four_images, sample_four_vectors = images[indices_four], vectors[indices_four]
    plt.figure()
    plt.axis('off')
    plt.subplot(2, 3, 1)
    plt.imshow(sample_three_images[0], cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(sample_four_images[1], cmap="gray")
    plt.subplot(2, 3, 3)
    interpolation_vec = sample_three_vectors[0] + sample_four_vectors[1]
    interpolated_img = model(interpolation_vec[np.newaxis, ...]).numpy().reshape(28, 28)
    plt.imshow(interpolated_img, cmap="gray")

    plt.subplot(2, 3, 4)
    plt.imshow(sample_four_images[0], cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(sample_three_images[1], cmap="gray")
    plt.subplot(2, 3, 6)
    interpolation_vec = sample_four_vectors[0] + sample_three_vectors[1]
    interpolated_img = model(interpolation_vec[np.newaxis, ...]).numpy().reshape(28, 28)
    plt.imshow(interpolated_img, cmap="gray")
    plt.suptitle("Interpolating different digit - Addition (Left + Middle = Right)")
    plt.savefig("GLO_interpolated_different_digits.png")
    plt.show()

if __name__ == '__main__':
    # q1_autoencoder()
    # q2_autoencoder()
    q3_gan()
    # q4_glo()
