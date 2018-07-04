import numpy as np

epsilon = 0.0000001


def linear_model(X, W, b):
    return np.matmul(X, W) + b


def sigmoid_stable(x):
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def sigmoid(g):
    return np.vectorize(sigmoid_stable, otypes=[np.float32])(g)


def logistic_model(X, W, b):
    return sigmoid(linear_model(X, W, b))


def softmax(X):
    exp_x = np.exp(X)
    return exp_x / exp_x.sum(axis=1).reshape((exp_x.shape[0], 1))


def softmax_regression_model(X, W, b):
    return softmax(linear_model(X, W, b))


def mse_cost(h, y):
    diff = h - y
    return 0.5 * np.matmul(diff.transpose(), diff) / y.shape[0]


def mse_cost_dev(X, y, h):
    diff = h - y
    return (np.matmul(X.transpose(), diff) / y.shape[0]), diff.mean(axis=0).reshape((1, y.shape[1]))


def log_cost(h, y):
    return -(y * np.log(h + epsilon) + (1.0 - y) * np.log(1.0 - h + epsilon)).mean()


def log_cost_dev(X, y, h):
    diff = h - y
    return (np.matmul(X.transpose(), diff) / y.shape[0]), diff.mean(axis=0).reshape((1, y.shape[1]))


def crossentropy_cost(h, y):
    return -(y * np.log(h + epsilon)).mean()


def crossentropy_cost_dev(X, y, h):
    diff = h - y
    return (np.matmul(X.transpose(), diff) / y.shape[0]), diff.mean(axis=0).reshape((1, y.shape[1]))


def gd_update(W, b, d_W, d_b, lr=0.01):
    return (W - lr * d_W), (b - lr * d_b)


def create_parameters(feature_size, class_cnt=1):
    param_W = np.random.randn(
        feature_size * class_cnt).reshape((feature_size, class_cnt))
    param_b = np.random.randn(class_cnt).reshape(1, class_cnt)
    return param_W, param_b


def std_normalize(X):
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    for col in range(0, X.shape[1]):
        X[:, col] = (X[:, col] - means[col]) / stds[col]
    return stds, means


def data_normalize(X, stds, means):
    for col in range(0, X.shape[1]):
        X[:, col] = (X[:, col] - means[col]) / stds[col]


def binary_accuracy(h, y, threshold=0.5):
    right_cnt = 0
    for cid in range(0, y.shape[0]):
        if (y[cid][0] > 0.5) == (h[cid][0] > threshold):
            right_cnt += 1
    return right_cnt / y.shape[0]


def binary_confusion_matrix(h, y, threshold=0.5):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for cid in range(0, y.shape[0]):
        if h[cid][0] > threshold:
            if y[cid][0] > 0.5:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if y[cid][0] > 0.5:
                false_neg += 1
            else:
                true_neg += 1
    pred_pos = true_pos + false_pos
    total_pos = true_pos + false_neg
    pc = 0 if pred_pos == 0 else (true_pos / pred_pos)
    rc = 0 if total_pos == 0 else (true_pos / total_pos)
    f1 = 0 if (pc + rc) == 0 else (2.0 * (pc * rc) / (pc + rc))
    return pc, rc, f1


def categorical_accuracy(h, y):
    mh = np.argmax(h, axis=1)
    my = np.argmax(y, axis=1)
    cnt = mh.shape[0]
    acc = 0
    for i in range(0, cnt):
        if mh[i] == my[i]:
            acc += 1
    return acc / cnt
