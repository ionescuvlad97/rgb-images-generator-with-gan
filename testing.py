import os
import time
import pickle
import math
import random

# Third Party Libraries
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from utils import train_test_split
from utils import data_generator
from utils import show_images
from utils import get_lightness
from utils import convert_rgb2lab
from utils import convert_lab2rgb
from utils import define_generator_v1
from utils import define_generator_v2
from utils import plot_results
from utils import create_testing_directory_tree
from utils import save_np_train_images
from utils import get_one_example_per_class
from utils import calculate_inception_score
from utils import generator_loss

# ---------------

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_all_models_names():
    path = 'models'
    db_names = []
    models_paths = []
    for db_name in os.listdir(path):
        for model_name in os.listdir(os.path.join(path, db_name)):
            db_names.append(db_name)
            models_paths.append(os.path.join(path, db_name, model_name))
    return db_names, models_paths


def predict_generated_images(model_path, db_name, batch_size=100, overwrite=False):
    if overwrite:
        model_version = int(model_path[model_path.find('v')+1: model_path.find('v')+2])
        image_size = int(model_path[model_path.find('is')+2: model_path.find('is')+4])

        input_shape_generator = (image_size, image_size, 1)

        generator = define_generator_v1(input_shape_generator) if model_version == 1 \
            else define_generator_v2(input_shape_generator)
        generator.load_weights(model_path)

        with open('np_test_images/test_images_{}.npy'.format(image_size), 'rb') as f:
            test_images = np.load(f)

        test_images_lab = np.array(convert_rgb2lab(test_images[:20000]))
        test_images_lightness = np.array(get_lightness(test_images_lab))
        test_images_gen = []
        for i in range(0, len(test_images_lightness), batch_size):
            test_images_gen.extend(generator(test_images_lightness[i:i+batch_size]))
        test_images_gen = np.array(test_images_gen)
        np.save("testing_results/{}/np_images/test_images_gen_lab_{}.npy".format(db_name, image_size),
                test_images_gen)

        test_images_gen_rgb = convert_lab2rgb(test_images_gen)
        np.save("testing_results/{}/np_images/test_images_gen_rgb_{}.npy".format(db_name, image_size),
                test_images_gen_rgb)
    else:
        print("Predicted images are already saved. If you want to predict them again set overwrite=True.\n")


def save_predicted_images(model_path, db_name, overwrite=False):
    if overwrite:
        image_size = int(model_path[model_path.find('is') + 2: model_path.find('is') + 4])

        with open('np_test_images/test_images_{}.npy'.format(image_size), 'rb') as f:
            test_images = np.load(f)

        with open('testing_results/{}/np_images/test_images_gen_rgb_{}.npy'.format(db_name, image_size), 'rb') as f:
            test_images_gen_rgb = np.load(f)

        test_images = list(test_images)
        random.Random(42).shuffle(test_images)

        test_images_lab = np.array(convert_rgb2lab(test_images[:100]))
        test_images_lightness = np.array(get_lightness(test_images_lab))

        for i in range(0, 25, 5):
            plot_results(test_images_lightness[i: i+5],
                         test_images[i: i+5],
                         test_images_gen_rgb[i: i+5],
                         'testing_results/{}/generated_images/test_gen_img_{}.png'.format(db_name, i // 5), n=5)
    else:
        print("The images are already plotted and saved. If you want to plot them again set overwrite=True.\n")


def calculate_mse(test_images, generated_images, batch_size=100):
    result = 0
    for i in range(0, len(test_images), batch_size):
        mse = generator_loss(generated_images[i: i+batch_size], test_images[i: i+batch_size])
        result += mse
    return result / (len(test_images) / batch_size)


def calculate_psnr(test_images, generated_images, batch_size=100):
    result = 0
    for i in range(0, len(test_images), batch_size):
        psnr = tf.image.psnr(test_images[i: i+batch_size], generated_images[i: i+batch_size], max_val=255)
        p = sum(psnr) / len(psnr)
        result += p
    return result / (len(test_images) / batch_size)


def calculate_ssim(test_images, generated_images, batch_size=50):
    result = 0
    for i in range(0, len(test_images), batch_size):
        im1 = tf.image.convert_image_dtype(test_images[i: i+batch_size], tf.float32)
        im2 = tf.image.convert_image_dtype(generated_images[i: i+batch_size], tf.float32)
        ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                              filter_sigma=1.5, k1=0.01, k2=0.03)
        s = sum(ssim1) / len(ssim1)
        result += s
    return result / (len(test_images) / batch_size)


def calculate_colorfulness(images):
    result = []
    for image in images:
        img_L = image[:, :, 0]
        img_a = image[:, :, 1]
        img_b = image[:, :, 2]

        img_L_2 = np.power(img_L, 2)
        img_a_2 = np.power(img_a, 2)
        img_b_2 = np.power(img_b, 2)

        numerator = sum(map(sum, img_a_2 + img_b_2))
        denominator = sum(map(sum, img_L_2 + img_a_2 + img_b_2))

        s_ab = np.sqrt(np.divide(numerator, denominator)) * 100
        result.append(s_ab)

    return sum(result) / len(result)


def calculate_metrics(model_path, db_name):
    image_size = int(model_path[model_path.find('is') + 2: model_path.find('is') + 4])

    with open("np_test_images/test_images_{}.npy".format(image_size), "rb") as f:
        test_images = np.load(f)
    with open("testing_results/{}/np_images/test_images_gen_rgb_{}.npy".format(db_name, image_size), "rb") as f:
        test_images_gen_rgb = np.load(f)
    test_images = test_images[:20000]
    test_images_gen_rgb = test_images_gen_rgb
    print(test_images.shape)
    print(test_images_gen_rgb.shape)

    mse = calculate_mse(test_images, test_images_gen_rgb)
    psnr = calculate_psnr(test_images, test_images_gen_rgb)
    ssim = calculate_ssim(test_images, test_images_gen_rgb)
    s_ab_1 = calculate_colorfulness(test_images)
    s_ab_2 = calculate_colorfulness(test_images_gen_rgb)

    print("MSE: {}".format(mse))
    print("PSNR: {}".format(psnr))
    print("SSIM: {}".format(ssim))
    print("S_ab real images: {}".format(s_ab_1))
    print("S_ab generated images: {}".format(s_ab_2))

    with open("testing_results/{}/metrics/testing_metrics.txt".format(db_name), "w") as f:
        f.write("MSE: {}".format(mse) + "\n")
        f.write("PSNR: {}".format(psnr) + "\n")
        f.write("SSIM: {}".format(ssim) + "\n")
        f.write("S_ab real images: {}".format(s_ab_1) + "\n")
        f.write("S_ab generated images: {}".format(s_ab_2) + "\n")


def main():
    dataset_path = r"D:\Datasets\fruits-360\Test"
    image_size = 64
    save_np_train_images(dataset_path, "places", image_size, overwrite=True)
    db_names = [
        "db_name_places_v1",
        "db_name_places_v2"
    ]
    models_paths = [
        "places_model_v1",
        "places_model_V2"
    ]
    # db_names, models_paths = get_all_models_names()
    print(db_names)
    print(models_paths)
    for db_name in db_names:
        create_testing_directory_tree(db_name)

    for model_path, db_name in zip(models_paths, db_names):
        print(model_path)
        print(db_name)
        try:
            predict_generated_images(model_path, db_name)
            save_predicted_images(model_path, db_name)
            calculate_metrics(model_path, db_name)
        except ValueError as e:
            print(e)


if __name__ == '__main__':
    main()
    # test()
