import numpy as np

"""
Do not modify this file.
"""

# Softmax loss and Softmax gradient
### Loss functions ###

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST fashion_mnist_data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def data_loader_mnist(data_dir):
    Xtrain, Ytrain = load_mnist(data_dir, kind='train')
    Xtest, Ytest = load_mnist(data_dir, kind='t10k')

    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    # sample 10k training, 2k validation
    label_idxs = [np.where(Ytrain == i)[0] for i in range(10)]
    Xtrain, Ytrain, Xvalid, Yvalid = \
        np.concatenate([Xtrain[label_idxs[i][:1000]] for i in range(10)]), \
        np.concatenate([Ytrain[label_idxs[i][:1000]] for i in range(10)]), \
        np.concatenate([Xtrain[label_idxs[i][-200:]] for i in range(10)]), \
        np.concatenate([Ytrain[label_idxs[i][-200:]] for i in range(10)])

    # shuffle data before training
    idx_order = np.random.permutation(Xtrain.shape[0])
    Xtrain = Xtrain[idx_order]
    Ytrain = Ytrain[idx_order]

    return Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest


class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output


def predict_label(f):
    # This is a function to determine the predicted label given scores
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY