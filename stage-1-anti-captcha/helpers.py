import os
import uuid
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha

corpus = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
corpus_len = len(corpus)

def gen_captcha_str(length):
    if(length < 1 or length > 6):
        return None
    cl = list(corpus)
    series = [random.choice(cl) for i in range(0, length)]
    return ''.join(series)

def batch_generate_captcha_mat(count, cnt=1, w=30, h=30):
    capt = ImageCaptcha(width=cnt * 34 + 26, height=60)
    mat_x = np.ndarray((count, w, h), dtype=np.uint8)
    mat_y = np.zeros((count, corpus_len), dtype=np.uint8)
    for i in range(0, count):
        cid = random.randint(0, corpus_len - 1)
        c = corpus[cid]
        capt_img = np.array(capt.generate_image(c))
        gray_img = cv2.cvtColor(capt_img, cv2.COLOR_RGB2GRAY)
        scle_img = cv2.resize(gray_img, (w, h))        
        mat_x[i] = scle_img
        mat_y[i][cid] = 1.0
    return mat_x, mat_y

def show_img(img, zoom=4, dpi=80):
    w = img.shape[0]
    h = img.shape[1]
    plt.figure(figsize=(w*zoom/dpi, h*zoom/dpi), dpi=dpi)
    plt.axis('off')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    return

def prepare_data(X, y):
    return (X / 255).astype(np.float32), y

def load_and_prepare_data(file):
    dt = np.load(file)
    dt_x = dt['X']
    dt_y = dt['y']
    return prepare_data(dt_x, dt_y)

def one_hot_to_label(h):
    return corpus[np.argmax(h)]

def showcase(X, y, h, case_num=10):
    for test_i in range(0, case_num):
        rid = random.randint(0, y.shape[0] - 1)
        show_img(X[rid])
        print('truth: {0}, h: {1}'.format(one_hot_to_label(y[rid]), one_hot_to_label(h[rid])))

def combine_feature(X, model1, model2, model3):
    f1 = model1.predict(X)
    f2 = model2.predict(X)
    f3 = model3.predict(X.reshape(X.shape[0], X.shape[1], X.shape[2], 1))
    comp = np.ndarray((X.shape[0], corpus_len*3), dtype=np.float32)
    comp[:, 0:corpus_len] = f1
    comp[:,corpus_len:corpus_len*2] = f2
    comp[:,corpus_len*2:corpus_len*3] = f3
    return comp


if __name__ == '__main__':
    
    print('generating training data...')
    train_X, train_y = batch_generate_captcha_mat(100000, cnt=1, w=30, h=30)
    print('training data shape X {0}, y {1}'.format(train_X.shape, train_y.shape))
    
    print('generating validation data...')
    validate_X, validate_y = batch_generate_captcha_mat(20000, cnt=1, w=30, h=30)
    print('validate data shape X {0}, y {1}'.format(validate_X.shape, validate_y.shape))
    
    print('generating testing data...')
    test_X, test_y = batch_generate_captcha_mat(10000, cnt=1, w=30, h=30)
    print('test data shape X {0}, y {1}'.format(test_X.shape, test_y.shape))

    print('save data to files...')
    np.savez_compressed('./data.npz', X=train_X, y=train_y)
    np.savez_compressed('./val.npz', X=validate_X, y=validate_y)
    np.savez_compressed('./test.npz', X=test_X, y=test_y)
    
    print('done!')
