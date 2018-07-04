# -*- coding:utf-8 -*-

import os
import cv2
import datetime
from keras import layers
from keras.models import Model, load_model
from keras.optimizers import SGD
from prepare_data import (
    data_generator,
    label_categorical_size
)

bn_axis = 3


def build_resnet_style_identity_block(filters, kernel_size, name, tensor):
    f1, f2, f3 = filters

    x = layers.Conv2D(f1, kernel_size=(1, 1), name='{}_conv_1'.format(name))(tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_1'.format(name))(x)
    x = layers.Activation('relu', name='{}_relu_1'.format(name))(x)

    x = layers.Conv2D(f2, kernel_size, padding='same', name='{}_conv_2'.format(name))(x)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_2'.format(name))(x)
    x = layers.Activation('relu', name='{}_relu_2'.format(name))(x)

    x = layers.Conv2D(f3, kernel_size=(1, 1), name='{}_conv_3'.format(name))(x)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_3'.format(name))(x)

    x = layers.add([x, tensor])
    x = layers.Activation('relu', name='{}_relu_add'.format(name))(x)

    return x


def build_resnet_style_conv_block(filters, kernel_size, name, tensor):
    f1, f2, f3 = filters

    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(2, 2), name='{}_conv_1'.format(name))(tensor)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_1'.format(name))(x)
    x = layers.Activation('relu', name='{}_relu_1'.format(name))(x)

    x = layers.Conv2D(f2, kernel_size, padding='same', name='{}_conv_2'.format(name))(x)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_2'.format(name))(x)
    x = layers.Activation('relu', name='{}_relu_2'.format(name))(x)

    x = layers.Conv2D(f3, kernel_size=(1, 1), name='{}_conv_3'.format(name))(x)
    x = layers.BatchNormalization(axis=bn_axis, name='{}_bn_3'.format(name))(x)

    y = layers.Conv2D(f3, kernel_size=(1, 1), strides=(2, 2), name='{}_conv_shortcut'.format(name))(tensor)
    y = layers.BatchNormalization(axis=bn_axis, name='{}_bn_shortcut'.format(name))(y)

    x = layers.add([x, y])
    x = layers.Activation('relu', name='{}_relu_add'.format(name))(x)

    return x


def build_conv_block(filters, kernel_size, name, tensor):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', name='{}_conv'.format(name))(tensor)
    x = layers.BatchNormalization(name='{}_bn'.format(name))(x)
    x = layers.Activation('relu', name='{}_relu'.format(name))(x)
    x = layers.MaxPooling2D(name='{}_pooling'.format(name))(x)
    return x


def build_classify_node(tensor, class_count, kernel_size, name):
    x = layers.Conv2D(class_count, kernel_size=kernel_size, strides=(1, 1), padding='valid',
                      activation='softmax', name='{}_conv'.format(name))(tensor)
    x = layers.Flatten(name='{}_flatten'.format(name))(x)
    return x


def build_conv_multi_heads_model(class_count, input_shape):
    inputs = layers.Input(shape=input_shape)

    x = build_conv_block(64,  (3, 3), 'block1', inputs)
    x = build_conv_block(128, (3, 3), 'block2', x)
    x = build_conv_block(256, (3, 3), 'block3', x)
    x = build_conv_block(64,  (1, 1), 'bottleneck', x)

    bottle_shape = (int(x.shape[1]), int(x.shape[2]))
    x = [build_classify_node(x, class_count, kernel_size=bottle_shape, name='char_1'),
         build_classify_node(x, class_count, kernel_size=bottle_shape, name='char_2'),
         build_classify_node(x, class_count, kernel_size=bottle_shape, name='char_3'),
         build_classify_node(x, class_count, kernel_size=bottle_shape, name='char_4')]

    model = Model(inputs=inputs, outputs=x, name='captcha_model')

    return model


def build_resnet_style_model(class_count, input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = build_resnet_style_conv_block([64, 64, 256], 3, 'res_1a', x)
    x = build_resnet_style_identity_block([64, 64, 256], 3, 'res_1b', x)
    x = build_resnet_style_identity_block([64, 64, 256], 3, 'res_1c', x)

    x = build_resnet_style_conv_block([128, 128, 512], 3, 'res_2a', x)
    x = build_resnet_style_identity_block([128, 128, 512], 3, 'res_2b', x)
    x = build_resnet_style_identity_block([128, 128, 512], 3, 'res_2c', x)
    x = build_resnet_style_identity_block([128, 128, 512], 3, 'res_2d', x)

    x = layers.Conv2D(256, kernel_size=(1, 1), padding='valid', name='bottleneck_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bottleneck_bn')(x)
    x = layers.Activation('relu', name='bottleneck_relu')(x)

    bottle_shape = (int(x.shape[1]), int(x.shape[2]))
    x = layers.AveragePooling2D(pool_size=bottle_shape, name='avg_pool')(x)
    x = [build_classify_node(x, class_count, kernel_size=(1, 1), name='char_1'),
         build_classify_node(x, class_count, kernel_size=(1, 1), name='char_2'),
         build_classify_node(x, class_count, kernel_size=(1, 1), name='char_3'),
         build_classify_node(x, class_count, kernel_size=(1, 1), name='char_4')]

    model = Model(inputs=inputs, outputs=x, name='captcha_model')

    return model


def train_model(training_items=80000, validation_items=20000, batch_size=64,
                epochs=20, init_epoch=0, check_point=None, fine_tune=False, force_compile=False):
    model_name = 'captcha_model_{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

    train_gen = data_generator('../data/training/', batch_size=batch_size)
    val_gen = data_generator('../data/validation/', batch_size=batch_size)

    if check_point is None:
        model = build_resnet_style_model(label_categorical_size, input_shape=(112, 112, 3))
    else:
        model = load_model(check_point)

    print(model.summary())

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    optimizer = sgd if fine_tune else 'adam'

    if check_point is None or force_compile:
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.fit_generator(train_gen,
                        steps_per_epoch=training_items // batch_size,
                        validation_data=val_gen,
                        validation_steps=validation_items // batch_size,
                        epochs=epochs,
                        initial_epoch=init_epoch)

    model.save('../models/{}.model'.format(model_name), include_optimizer=True)
    model.save('../models/{}.checkpoint'.format(model_name), include_optimizer=False)


if __name__ == '__main__':
    train_model(check_point='../models/captcha_model_20180705004403.checkpoint',
                init_epoch=20,
                epochs=50,
                fine_tune=True,
                force_compile=True)
