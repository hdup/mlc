# -*- coding:utf-8 -*-

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy


def segmented_categorical_crossentropy(y_true, y_pred):
    return (categorical_crossentropy(y_true[:, 0:36], y_pred[:, 0:36]) +
            categorical_crossentropy(y_true[:, 36:72], y_pred[:, 36:72]) +
            categorical_crossentropy(y_true[:, 72:108], y_pred[:, 72:108]) +
            categorical_crossentropy(y_true[:, 108:144], y_pred[:, 108:144])) / 4.0


def segmented_categorical_accuracy(y_true, y_pred):
    return (categorical_accuracy(y_true[:, 0:36], y_pred[:, 0:36]) +
            categorical_accuracy(y_true[:, 36:72], y_pred[:, 36:72]) +
            categorical_accuracy(y_true[:, 72:108], y_pred[:, 72:108]) +
            categorical_accuracy(y_true[:, 108:144], y_pred[:, 108:144])) / 4.0
