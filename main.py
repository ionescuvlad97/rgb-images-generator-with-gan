# Standard Libraries
import os
import time

# Third Party Libraries
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

# Custom Libraries
from utils import show_images
from utils import train_test_split
from utils import data_generator
from utils import convert_lab2rgb
from utils import convert_rgb2lab
from utils import get_lightness
from utils import define_discriminator
from utils import define_generator_v1
from utils import define_generator_v2
from utils import train_step
from utils import save_plot
from utils import plot_results
from utils import save_model
from utils import get_reference_images
from utils import create_directory_tree
from utils import calculate_inception_score
from utils import save_losses
from utils import save_parameters

# Global Variables
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # -------------- Variables -------------- #
    dataset_path = r"D:\Datasets\fruits-360\Training"
    reference_images_path = r"D:\Datasets\fruits-360\Test"
    model_version = 2
    train_test_split_percent = 0.8
    batch_size = 64 # 128
    reference_images_batch_size = 25
    training_iterations_per_epoch = 200
    num_epochs = 25
    image_size = 64 # 32

    database_name = 'fruits_v{}_is{}_e{}_bs{}_it{}'.format(model_version,
                                                           image_size,
                                                           num_epochs,
                                                           batch_size,
                                                           training_iterations_per_epoch)

    create_directory_tree(database_name)

    training_path_list, testing_path_list = train_test_split(dataset_path, train_test_split_percent)
    print(len(training_path_list))

    input_shape_generator = (image_size, image_size, 1)
    input_shape_discriminator = (image_size, image_size, 3)

    reference_images = get_reference_images(reference_images_path, reference_images_batch_size, image_size)
    show_images(reference_images, 5, 5)
    filename = 'results/{}/generated_images/reference_images.png'.format(database_name)
    plt.savefig(filename)
    plt.close()

    generator_optimizer = tf.keras.optimizers.Adam(0.0005)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)

    # -------------- Models -------------- #
    # Select generator model - v1 or v2
    generator = define_generator_v1(input_shape_generator)
    discriminator = define_discriminator(input_shape_discriminator)

    gen_loss = []
    disc_loss = []
    ic_score = []
    ic_score_std = []
    time_list = []

    start_train_time = time.time()
    # -------------- Training -------------- #
    for e in range(num_epochs):
        b = 1
        iterations_count = 0
        sum_gen_loss = 0
        sum_disc_loss = 0
        sum_ic_score = 0
        sum_ic_score = 0
        start_epoch_time = time.time()

        for data in data_generator(training_path_list, batch_size, image_size):
            y = np.array(convert_rgb2lab(data))
            x = np.array(get_lightness(y))
            # Here ( x , y ) represents a batch from our training dataset.
            gen_loss_batch, disc_loss_batch = train_step(x, y,
                                                         generator,
                                                         discriminator,
                                                         generator_optimizer,
                                                         discriminator_optimizer, e + 1, b)
            b += 1
            iterations_count += 1
            sum_gen_loss += gen_loss_batch
            sum_disc_loss += disc_loss_batch

            if iterations_count == training_iterations_per_epoch:
                break

        testing_images_gray = np.array(get_lightness(convert_rgb2lab(reference_images)))
        testing_images_generated = generator(testing_images_gray).numpy()
        testing_images_generated = np.array(convert_lab2rgb(testing_images_generated))
        image_path = 'results/{}/generated_images/generated_plot_e{:03d}.png'.format(database_name, e + 1)
        # save_plot(testing_images, image_path, 5)
        plot_results(testing_images_gray, reference_images, testing_images_generated, image_path, n=5)
        save_model(generator, image_size, model_version, database_name)

        avg_gen_loss = sum_gen_loss / training_iterations_per_epoch
        avg_disc_loss = sum_disc_loss / training_iterations_per_epoch
        gen_loss.append(avg_gen_loss)
        disc_loss.append(avg_disc_loss)
        total_epoch_time = time.time() - start_epoch_time
        time_list.append(total_epoch_time)
        print("Epoch: {}, Average Generator Loss: {:.2f}, Average Discriminator Loss {:.2f}, Epoch time: {:.2f}s"
              .format(e + 1,
                      avg_gen_loss,
                      avg_disc_loss,
                      total_epoch_time))

    avg_time = sum(time_list)/len(time_list)
    total_training_time = time.time() - start_train_time

    plt.figure(1)
    plt.plot(range(1, num_epochs + 1), gen_loss)
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = 'results/{}/metrics/generator_loss.png'.format(database_name)
    plt.savefig(filename)
    plt.close()
    plt.figure(2)
    plt.plot(range(1, num_epochs + 1), disc_loss)
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = 'results/{}/metrics/discriminator_loss.png'.format(database_name)
    plt.savefig(filename)
    plt.close()
    plt.show()

    save_parameters(database_name,
                    model_version,
                    batch_size,
                    training_iterations_per_epoch,
                    num_epochs,
                    image_size,
                    gen_loss[-1],
                    disc_loss[-1],
                    total_training_time,
                    avg_time)

    save_losses(database_name, 'gen_loss', gen_loss)
    save_losses(database_name, 'disc_loss', disc_loss)

    print(gen_loss)
    print(disc_loss)


if __name__ == "__main__":
    main()
