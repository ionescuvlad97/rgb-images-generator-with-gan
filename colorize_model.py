# Standard Libraries
import ssl

# Third Party Libraries
import numpy as np
from numpy.random import rand
from numpy.random import randint
import pandas as pd
from matplotlib import pyplot as plt
from skimage import color
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.cifar10 import load_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras import Model
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.utils.vis_utils import plot_model

# Custom Libraries

# Eliminate SSL certificate in order to download the database

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def show_images(images, length, width):
    for i in range(length*width):
        plt.subplot(length, width, 1+i)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()


def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def define_generator():

    # Encoder
    encoder_input = Input(shape=(32, 32, 1))
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(encoder_input)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoded = Conv2D(filters=2, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    concat = Concatenate()([encoder_input, decoded])

    autoencoder = Model(encoder_input, concat)

    return autoencoder


def define_gan(g_model, d_model):
    d_model.trainable = False

    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


def load_real_samples():
    (train_x, _), (_, _) = load_data()
    X = train_x.astype("float32")
    X = X/255.0

    return X


def generate_real_samples(data_set, n_samples):
    ix = randint(0, data_set.shape[0], n_samples)
    X = data_set[ix]
    y = np.ones((n_samples, 1))

    return X, y


def generate_fake_samples(model, real_samples, n_samples):
    real_samples_lightness = np.array([color.rgb2lab(sample)[:, :, 0] for sample in real_samples])
    X = np.array([model.predict(np.expand_dims(sample, axis=0)) for sample in real_samples_lightness])
    X = np.squeeze(X, axis=1)
    X = np.array([color.lab2rgb(sample) for sample in X])
    y = np.zeros((n_samples, 1))
    return X, y


# def down(filters, kernel_size):
#     downsample = Sequential()
#     downsample.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=2))
#     downsample.add(LeakyReLU(alpha=0.2))
#
#     return downsample
#
#
# def up(filters, kernel_size):
#     upsample = Sequential()
#     upsample.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=2))
#     upsample.add(LeakyReLU(alpha=0.2))
#
#     return upsample


def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, g_model, d_model, data_set, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(data_set, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, X_real, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, data_set, n_epochs=100, n_batch=256):
    batch_per_epo = int(data_set.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(batch_per_epo):
            X_real, y_real = generate_real_samples(data_set, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, X_real, half_batch)

            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            d_loss, _ = d_model.train_on_batch(X, y)

            X_gan, _ = generate_real_samples(data_set, n_batch)

            X_gan = np.array([color.rgb2lab(sample)[:, :, 0] for sample in X_gan])
            y_gan = np.ones((n_batch, 1))

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, batch_per_epo, d_loss, g_loss))
        if (i + 1) % 1 == 0:
            summarize_performance(i, g_model, d_model, data_set)


# if __name__ == "__main__":
dataset = load_real_samples()
print(dataset.shape)
number_of_samples = 25

discriminator = define_discriminator()
generator = define_generator()
gan = define_gan(generator, discriminator)

discriminator.summary()
generator.summary()
gan.summary()

train(generator, discriminator, gan, dataset)
