# -*- coding:utf8 -*-

import os
import cv2
import numpy as np
from keras.utils import to_categorical
from shutil import copyfile


# you can resize or not
image_resized_shape = (112, 112)
label_chars = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
label_categorical_size = len(label_chars)


def label_to_one_hot(ch):
    index = label_chars.find(ch)
    return to_categorical(index, label_categorical_size)


def preprocess_image(img_file, dsize):
    # NOTE, cv2 reads image with channel order: BGR
    img_data = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img_data = cv2.resize(img_data, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    img_data = img_data / 255.
    # load label
    label = img_file[-8:-4]
    return (img_data.astype(np.float32),
            label_to_one_hot(label[0]), label_to_one_hot(label[1]),
            label_to_one_hot(label[2]), label_to_one_hot(label[3]))


def convert_images_to_np(image_folder, np_file):
    tuples = [preprocess_image(os.path.join(image_folder, f), dsize=image_resized_shape)
              for f in os.listdir(image_folder) if f.endswith('.png')]

    data_img = np.stack([t[0] for t in tuples], axis=0)
    data_y0 = np.stack([t[1] for t in tuples], axis=0)
    data_y1 = np.stack([t[2] for t in tuples], axis=0)
    data_y2 = np.stack([t[3] for t in tuples], axis=0)
    data_y3 = np.stack([t[4] for t in tuples], axis=0)

    np.save(np_file, (data_img, data_y0, data_y1, data_y2, data_y3))


def split_training_data(image_folder, root_folder, portion=0.8):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    total_files = len(image_files)
    train_folder = os.path.join(root_folder, 'training')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    validation_folder = os.path.join(root_folder, 'validation')
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)
    for i, f in enumerate(image_files):
        if float(i) / total_files < portion:
            copyfile(os.path.join(image_folder, f), os.path.join(train_folder, f))
        else:
            copyfile(os.path.join(image_folder, f), os.path.join(validation_folder, f))


def data_generator(image_folder, batch_size=32):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    data = []
    data_y0 = []
    data_y1 = []
    data_y2 = []
    data_y3 = []
    while True:
        for im_file in image_files:
            im_data, y0, y1, y2, y3 = preprocess_image(im_file, dsize=image_resized_shape)
            data.append(im_data)
            data_y0.append(y0)
            data_y1.append(y1)
            data_y2.append(y2)
            data_y3.append(y3)
            if len(data) == batch_size:
                yield np.stack(data, axis=0), [np.stack(data_y0, axis=0), np.stack(data_y1, axis=0),
                                               np.stack(data_y2, axis=0), np.stack(data_y3, axis=0)]
                data = []
                data_y0 = []
                data_y1 = []
                data_y2 = []
                data_y3 = []


if __name__ == '__main__':
    split_training_data('../data/images', '../data', portion=0.8)
