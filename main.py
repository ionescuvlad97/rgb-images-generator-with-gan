# Standard Libraries

# Third Party Libraries
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Custom Libraries
from utils import show_images
from utils import train_test_split
from utils import data_generator
from utils import convert_lab2rgb
from utils import convert_rgb2lab
from utils import get_lightness
from utils import define_discriminator
from utils import define_generator
from utils import train_step
from utils import save_plot
from utils import save_model
from utils import get_reference_images

# Global Variables
tf.config.run_functions_eagerly(True)


def main():
    dataset_path = r"E:\Politehnica\Master\Disertatie\Datasets\fruits-360_dataset\fruits-360\Training"
    reference_images_path = r"E:\Politehnica\Master\Disertatie\Datasets\fruits-360_dataset\fruits-360\Test"
    train_test_split_percent = 0.8
    training_path_list, testing_path_list = train_test_split(dataset_path, train_test_split_percent)
    model_version = 0
    batch_size = 256
    reference_images_batch_size = 25
    training_iterations_per_epoch = 3
    num_epochs = 3
    image_size = 32
    input_shape_generator = (image_size, image_size, 1)
    input_shape_discriminator = (image_size, image_size, 3)

    reference_images = get_reference_images(reference_images_path, reference_images_batch_size, image_size)
    show_images(reference_images, 5, 5)
    filename = 'generated_images/reference_images.png'
    plt.savefig(filename)
    plt.close()

    generator_optimizer = tf.keras.optimizers.Adam(0.0005)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)

    generator = define_generator(input_shape_generator)
    discriminator = define_discriminator(input_shape_discriminator)

    gen_loss = []
    disc_loss = []

    for e in range(num_epochs):
        b = 1
        iterations_count = 0
        sum_gen_loss = 0
        sum_disc_loss = 0
        for data in data_generator(training_path_list, batch_size, image_size):
            y = np.array(convert_rgb2lab(data))
            x = np.array(get_lightness(y))
            # Here ( x , y ) represents a batch from our training dataset.
            gen_loss_batch, disc_loss_batch = train_step(x, y,
                                                         generator,
                                                         discriminator,
                                                         generator_optimizer,
                                                         discriminator_optimizer, e+1, b)
            b += 1
            iterations_count += 1
            sum_gen_loss += gen_loss_batch
            sum_disc_loss += disc_loss_batch
            if iterations_count == training_iterations_per_epoch:
                break
        avg_gen_loss = sum_gen_loss/training_iterations_per_epoch
        avg_disc_loss = sum_disc_loss/training_iterations_per_epoch
        gen_loss.append(avg_gen_loss)
        disc_loss.append(avg_disc_loss)

        print("Epoch: {}, Average Generator Loss: {:.2f}, Average Discriminator Loss {:.2f}".format(e,
                                                                                                    avg_gen_loss,
                                                                                                    avg_disc_loss))

        testing_images = np.array(get_lightness(convert_rgb2lab(reference_images)))
        testing_images = generator(testing_images).numpy()
        testing_images = np.array(convert_lab2rgb(testing_images))
        save_plot(testing_images, e, 5)
        save_model(generator, image_size, model_version)

    plt.figure(1)
    plt.plot(range(1, num_epochs+1), gen_loss)
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = 'generated_images/generator_loss.png'
    plt.savefig(filename)
    plt.close()
    plt.figure(2)
    plt.plot(range(1, num_epochs+1), disc_loss)
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = 'generated_images/discriminator_loss.png'
    plt.savefig(filename)
    plt.close()
    plt.show()

    print(gen_loss)
    print(disc_loss)


def test():
    dataset_path = r"E:\Politehnica\Master\Disertatie\Datasets\fruits-360_dataset\fruits-360\Test"
    train_test_split_percent = 0.8
    training_path_list, testing_path_list = train_test_split(dataset_path, train_test_split_percent)
    model_version = 0
    batch_size = 256
    testing_batch_size = 25
    training_iterations_per_epoch = 4
    num_epochs = 20
    image_size = 32
    input_shape_generator = (image_size, image_size, 1)
    input_shape_discriminator = (image_size, image_size, 3)

    test_img = get_test_batch(dataset_path, testing_batch_size, image_size)

    testing_generator = data_generator(testing_path_list, testing_batch_size, image_size)
    testing_batch = next(testing_generator)

    print(np.shape(test_img))

    save_plot(test_img, "", 5)

    generator = define_generator(input_shape_generator)
    discriminator = define_discriminator(input_shape_discriminator)

    print(generator.summary())
    print(discriminator.summary())


def test_1():
    reference_images_path = r"E:\Politehnica\Master\Disertatie\Datasets\fruits-360_dataset\fruits-360\Test"
    reference_images_batch_size = 25
    image_size = 32

    reference_images = get_reference_images(reference_images_path, reference_images_batch_size, image_size)

    show_images(reference_images, 5, 5)
    plt.show()


if __name__ == "__main__":
    main()
    # test()
    # test_1()
