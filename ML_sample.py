from utils import softmax_cross_entropy, data_loader_mnist, predict_label, DataSplit
from tqdm import tqdm
from copy import deepcopy
import argparse
import json
import time
import numpy as np


###################################
#   Only modify the TODO blocks   #
###################################


# 1. One linear Neural Network layer with forward and backward steps
class linear_layer:
    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """
    def __init__(self, input_D, output_D):
        self.params = dict()
        self.gradient = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))
                           
    def forward(self, X):
        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (N is the batch size)

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """
        forward_output = np.dot(X, self.params['W']) + self.params['b']

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivative of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'],
            and store them in self.gradient['W'], self.gradient['b'].

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss w.r.t. X[i].
        """
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = grad.mean(axis=0) * X.shape[0]
        backward_output = np.dot(grad, self.params['W'].T)

        return backward_output


# 2. ReLU Activation
class relu:
    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """
        forward_output = np.maximum(0, X)

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """
        X = X > 0
        backward_output = grad * X

        return backward_output


# 3. Mini-batch Gradient Descent Optimization
def miniBatchGradientDescent(model, _learning_rate):
    for module_name, module in model.items():

        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                # This is the gradient for the parameter named "key" in this module
                g = module.gradient[key]
                w = module.params[key]
                w -= _learning_rate * g

    return model


### Model forward pass ###
def forward_pass(model, x, y):
    a1 = model['L1'].forward(x)  # output of the first linear layer
    h1 = model['nonlinear1'].forward(a1)  # output of ReLu
    a2 = model['L2'].forward(h1)  # output of the second linear layer
    loss = model['loss'].forward(a2, y)

    return a1, h1, a2, loss


# 4. Model backward pass
def backward_pass(model, x, a1, h1, a2, y):
    grad_a2 = model['loss'].backward(a2, y)
    grad_h1 = model['L2'].backward(h1, grad_a2)
    grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
    grad_x = model['L1'].backward(x, grad_a1)

### Compute the accuracy and loss of a model on some train/val/test dataset ###
def compute_accuracy_loss(N_data, DataSet, model, minibatch_size=1000):
    acc = 0.0
    loss = 0.0
    count = 0

    ### Iterate over the dataset ###
    for i in range(int(np.floor(N_data / minibatch_size))):
        x, y = DataSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

        ### Forward pass ###
        _, _, a2, batch_loss = forward_pass(model, x, y)
        loss += batch_loss
        acc += np.sum(predict_label(a2) == y)
        count += len(y)

    return acc / count, loss


### Use magnitude checker to ensure the SGD gradient is on the correct magnitude ###
def magnitude_checker(DataSet, model):
    ### Get 5 examples ###
    x, y = DataSet.get_example(np.arange(50))

    ### The L1-norm of the first layer gradient with batch size 50 ###
    a1, h1, a2, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, y)
    l1_norm_w_grad_five = model["L1"].gradient["W"].sum()

    ### Get 5k examples ###
    x, y = DataSet.get_example(np.arange(5000))

    ### The L1-norm of the first layer gradient with batch size 5k ###
    a1, h1, a2, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, y)
    l1_norm_w_grad_fivek = model["L1"].gradient["W"].sum()

    ### As unbiased estimations of the gradient, we expect them to have a similar magnitude ###
    print("Check the magnitude (L1-norm of layer L1) of gradient with batch size 50: %f and with batch size 5k: %f"
          % (l1_norm_w_grad_five, l1_norm_w_grad_fivek))


# 5. Use gradient checker to ensure the backprop implementation on SGD is correct ###
def gradient_checker(DataSet, model):
    ### Get the first example ###
    x, y = DataSet.get_example([0])

    ### Forward and backward of x, y ###
    a1, h1, a2, _ = forward_pass(model, x, y)
    backward_pass(model, x, a1, h1, a2, y)

    ### Gradients from the backpropagation: ###
    # the first dimension of L1 weight and bias, and the first dimension of L2 weight and bias
    grad_dict = {}
    grad_dict["L1_W_grad_first_dim"] = model['L1'].gradient["W"][0][0]
    grad_dict["L1_b_grad_first_dim"] = model['L1'].gradient["b"][0]
    grad_dict["L2_W_grad_first_dim"] = model['L2'].gradient["W"][0][0]
    grad_dict["L2_b_grad_first_dim"] = model['L2'].gradient["b"][0]

    ### Gradients from approximation ###
    # the first dimension of L1 weight and bias, and the first dimension of L2 weight and bias
    for name, grad in grad_dict.items():
        layer_name = name.split("_")[0]
        param_name = name.split("_")[1]

        ### Set a small epsilon ###
        epsilon_value = 1e-3
        epsilon = np.zeros(model[layer_name].params[param_name].shape)
        if len(epsilon.shape) == 2:
            epsilon[0][0] = epsilon_value
        else:
            epsilon[0] = epsilon_value

        model[layer_name].params[param_name] += epsilon
        _, _, _, f_w_add_epsilon = forward_pass(model, x, y)
        print(f_w_add_epsilon)
        model[layer_name].params[param_name] -= 2*epsilon
        _, _, _, f_w_minus_epsilon = forward_pass(model, x, y)
        print(f_w_minus_epsilon)
        model[layer_name].params[param_name] += epsilon
        approximate_gradient = (f_w_add_epsilon - f_w_minus_epsilon) / (2 * epsilon_value)

        print("Check the gradient of %s in the %s layer from backpropagation: %f and from approximation: %f"
              % (param_name, layer_name, grad, approximate_gradient))


def main(main_params):
    ### Set the random seed. DO NOT MODIFY. ###
    np.random.seed(int(main_params['random_seed']))

    ### Data processing ###
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_loader_mnist(data_dir=main_params['data_dir'])
N_train, d = Xtrain.shape
    N_val, _ = Xval.shape
    N_test, _ = Xtest.shape
    print("Training data size: %d, Validation data size: %d, Test data size: %d" % (N_train, N_val, N_test))

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)
    testSet = DataSplit(Xtest, Ytest)

    ### Building/defining MLP ###
    """
    In this script, we are going to build a MLP for a 10-class classification problem on Fashion MNIST.
    The network structure is input --> linear --> relu --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 128
    the output_layer size (num_L2) is 10
    """
    model = dict()
    num_L1 = 128
    num_L2 = 10

    ### Experimental setup ###
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])
    check_gradient = main_params['check_gradient']
    check_magnitude = main_params['check_magnitude']
    patience = int(main_params['early_stopping_patience'])

    ### Store evaluation results and best model
    train_acc_record = []
    train_loss_record = []
    val_acc_record = []
    val_loss_record = []
    best_epoch = 0
    best_model = None

    ### Optimization setting ###
    _learning_rate = float(main_params['learning_rate'])
    _step = 10

    ### Create objects (model modules) from the module classes ###
    model['L1'] = linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = relu()
    model['L2'] = linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = softmax_cross_entropy()

    ### Use magnitude checker to ensure the backprop implementation is on the correct magnitude ###
    if check_magnitude:
        magnitude_checker(trainSet, model)

    ### Use gradient checker to ensure the backprop implementation is correct ###
    if check_gradient:
        gradient_checker(trainSet, model)

    ### Set timer ###
    start_time = time.time()

    ### Run training and validation ###
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))

        idx_order = np.random.permutation(N_train)

        for i in tqdm(range(int(np.floor(N_train / minibatch_size)))):
            ### Get a mini-batch of data ###
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            ### Forward pass ###
            a1, h1, a2, _ = forward_pass(model, x, y)

            ### Backward pass ###
            backward_pass(model, x, a1, h1, a2, y)

            ### Gradient_update ###
            model = miniBatchGradientDescent(model, _learning_rate)

        ### Compute training accuracy and loss ###
        train_acc, train_loss = compute_accuracy_loss(N_train, trainSet, model)
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        ### Compute validation accuracy and loss ###
        val_acc, val_loss = compute_accuracy_loss(N_val, valSet, model)
        val_acc_record.append(val_acc)
        val_loss_record.append(val_loss)

        ### Print accuracy and loss ###
        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))
        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

        ### Update best model ###
        if val_acc == max(val_acc_record):
            best_model = deepcopy(model)
            best_epoch = t + 1
            patience = int(main_params['early_stopping_patience'])
        else:
            patience -= 1

        ### Earlystop ###
        if patience == 0:
            break

    ### Stop timer ###
    end_time = time.time()

    ### Compute test accuracy and loss ###
    test_acc, test_loss = compute_accuracy_loss(N_test, testSet, best_model)
    print('Test accuracy at the best epoch (epoch ' + str(best_epoch) + ') is ' + str(test_acc))

    ### Save file ###
    json.dump({'train': train_acc_record, 'val': val_acc_record, 'test': test_acc, 'time': end_time - start_time},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_b' + str(main_params['minibatch_size']) +
                   '.json', 'w'))

    print('Training time: ' + str(end_time - start_time))
    print('Finish running!')
    return train_loss_record, val_loss_record


def get_parser():
    ######################################################################################
    # These are the default arguments used to run your code.
    # You can modify them to test your code
    ######################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--data_dir', default='fashion_mnist_data')
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--num_epoch', default=200)
    parser.add_argument('--minibatch_size', default=1)
    parser.add_argument('--early_stopping_patience', default=3)
    parser.add_argument('--check_gradient', action="store_true", default=False,
                        help="Check the correctness of the gradient")
    parser.add_argument('--check_magnitude', action="store_true", default=False,
                        help="Check the magnitude of the gradient")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main_params = get_parser()
    main(main_params)
