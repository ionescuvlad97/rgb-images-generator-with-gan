# Standard Libraries
import random
from pathlib import Path
from PIL import Image
import os
import math

# Third Party Libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Custom Libraries

# Global Variables
tf.config.run_functions_eagerly(True)
mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy()


def create_directory_tree(database_name):
    Path("models/{}".format(database_name)).mkdir(parents=True, exist_ok=True)
    Path("results/{}/generated_images".format(database_name)).mkdir(parents=True, exist_ok=True)
    Path("results/{}/metrics".format(database_name)).mkdir(parents=True, exist_ok=True)


def show_images(images, length, width):
    for i in range(length * width):
        plt.subplot(length, width, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])


def train_test_split(path, train_size):
    dataset_paths_list = list(Path(path).rglob('*.jpg'))
    random.Random(42).shuffle(dataset_paths_list)
    threshold = int(train_size * len(dataset_paths_list))
    training_paths_list = dataset_paths_list[:threshold]
    testing_path_list = dataset_paths_list[threshold:]

    return training_paths_list, testing_path_list


def data_generator(path_list, b_size, new_size):
    result = []
    for image_path in path_list:
        image = plt.imread(image_path)
        image_resized = resize(image, (new_size, new_size), anti_aliasing=True)
        result.append(image_resized)
        if len(result) % b_size == 0:
            yield result
            result = []


def convert_rgb2lab(images):
    return [color.rgb2lab(image) for image in images]


def get_lightness(images):
    return [image[:, :, 0] for image in images]


def convert_lab2rgb(images):
    return [color.lab2rgb(image) for image in images]


def convert_tensor2image(images):
    return [np.asarray(Image.fromarray(image)) for image in images]


def define_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


def define_generator(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(16*4, kernel_size=(5, 5), strides=1)(inputs)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32*4, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32*4, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(32*4, kernel_size=(5, 5), strides=1)(conv1)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64*4, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64*4, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(64*4, kernel_size=(5, 5), strides=1)(conv2)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128*4, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128*4, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)

    bottleneck = tf.keras.layers.Conv2D(128*4, kernel_size=(3, 3), strides=1,
                                        activation='relu', padding='same')(conv3)

    concat_1 = tf.keras.layers.Concatenate()([bottleneck, conv3])
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128*4, kernel_size=(3, 3), strides=1, activation='relu')(concat_1)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128*4, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_3)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(64*4, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_3)

    concat_2 = tf.keras.layers.Concatenate()([conv_up_3, conv2])
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64*4, kernel_size=(3, 3), strides=1, activation='relu')(concat_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64*4, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(32*4, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_2)

    concat_3 = tf.keras.layers.Concatenate()([conv_up_2, conv1])
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32*4, kernel_size=(3, 3), strides=1, activation='relu')(concat_3)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_1)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(2, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_1)

    concat = Concatenate()([inputs, conv_up_1])

    autoencoder = Model(inputs, concat)

    return autoencoder


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape,
                                                                            maxval=0.1), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape,
                                                                             maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, real_y):
    real_y = tf.cast(real_y, 'float32')
    fake_output = tf.cast(fake_output, tf.float32)
    return mse(fake_output, real_y)


@tf.function
def train_step(input_x, real_y, generator, discriminator, generator_optimizer, discriminator_optimizer, e, b):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate an image -> G( x )
        generated_images = generator(input_x, training=True)
        # Probability that the given image is real -> D( x )
        real_output = discriminator(real_y, training=True)
        # Probability that the given image is the one generated -> D( G( x ) )
        generated_output = discriminator(generated_images, training=True)

        # L2 Loss -> || y - G(x) ||^2
        gen_loss = generator_loss(generated_images, real_y)
        # Log loss for the discriminator
        disc_loss = discriminator_loss(real_output, generated_output)

    print("Epoch: {}, Batch: {}, Generator Loss: {:.2f}, Discriminator Loss {:.2f}".format(
        e,
        b,
        gen_loss.numpy(),
        disc_loss.numpy()
    ))

    # Compute the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimize with Adam
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss.numpy(), disc_loss.numpy()


def save_plot(examples, path, n=10):
    # Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = path
    # filename = 'generated_images/example_images.png'
    plt.savefig(filename)
    plt.close()


def plot_results(gray_images, original_images, generated_images, path, n=3):
    cols = ['Grayscale Image', 'Original Image', 'Generated Image']

    fig, axes = plt.subplots(nrows=n, ncols=3)

    for i in range(n):
        axes[i][0].imshow(gray_images[i], cmap='gray')
        axes[i][1].imshow(original_images[i])
        axes[i][2].imshow(generated_images[i])

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    [ax.set_axis_off() for ax in axes.ravel()]
    plt.savefig(path)
    plt.close()


def save_model(model, image_size, model_version, database_name):
    filename = "models/{}/rgb_image_generator_gan_model_is{}_v{}.h5".format(database_name, image_size, model_version)
    model.save(filename)


def get_reference_images(path, b_size, new_size):
    result = []
    count = 0
    for d in os.listdir(path):
        for image_path in os.listdir(os.path.join(path, d)):
            image = plt.imread(os.path.join(path, d, image_path))
            image_resized = resize(image, (new_size, new_size), anti_aliasing=True)
            result.append(image_resized)
            break
        count += 1
        if count == b_size:
            break
    return result


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = math.floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std
