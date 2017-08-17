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


def softmax(X):
    exp_x = np.exp(X)
    return  exp_x / exp_x.sum()


def logistic_model(X, W, b):
    return sigmoid(linear_model(X, W, b))


def mse_cost(h, y):
    diff = h - y
    return 0.5 * np.matmul(diff.transpose(), diff).mean()


def mse_cost_dev(X, y, h):
    diff = h - y
    return (np.matmul(diff.transpose(), X) / X.shape[0]), (diff.mean())


def log_cost(h, y):
    return -(y * np.log(h + epsilon) + (1.0 - y) * np.log(1.0 - h + epsilon)).sum(axis=0) / y.shape[0]


def log_cost_dev(X, y, h):
    diff = h - y
    return ((diff * X).sum(axis=0) / y.shape[0]), (diff.sum(axis=0) / y.shape[0])


def gd_update(X, y, h, W, b, cost_dev_func, lr=0.01):
    d_W, d_b = cost_dev_func(X, y, h)
    return (W - lr * d_W), (b - lr * d_b)


def create_parameters(feature_size):
    param_W = np.random.randn(feature_size).reshape((feature_size, 1))
    param_b = np.random.randn()
    return param_W, param_b


def std_normalize(X):
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    for col in range(0, X.shape[1]):
        X[:, col] = (X[:, col] - means[col]) / stds[col]
    return stds, means

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
    pc = true_pos / (true_pos + false_pos)
    rc = true_pos / (true_pos + false_neg)
    f1 = 2.0 * (pc * rc) / (pc + rc)
    return pc, rc, f1
