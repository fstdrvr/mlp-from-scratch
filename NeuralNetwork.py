import copy
import sys
import numpy as np
import time

# Hyper-parameters
MU = 0 # Initial mean for weights assignment
SD = 0.1 # Initial standard deviation for weights assignment
LEARN_RATE = 0.03
BATCH_SIZE = 100
EPOCH = 25000
STOP = 10 # Early stopping counter

class Linear:
    """Fully connected linear layer"""
    def __init__(self, input_dimension, output_dimension):
        # Random initialization of weights and bias
        self.params = dict()
        self.params['W'] = np.random.normal(MU, SD, (input_dimension, output_dimension))
        self.params['b'] = np.random.normal(MU, SD, (1, output_dimension))
        # Partial derivatives of the loss function for this layer
        self.params['dW'] = np.zeros((input_dimension, output_dimension))
        self.params['db'] = np.zeros((1, output_dimension))

    def fwd_propagation(self, X):
        return np.dot(X, self.params['W']) + self.params['b']

    def bwd_propagation(self, X, dX):
        self.params['dW'] = np.dot(X.T, dX)
        self.params['db'] = dX.mean(axis=0) * X.shape[0]
        return np.dot(dX, self.params['W'].T)

class Relu:
    """Relu Activation layer"""
    def fwd_propagation(self, X):
        # Relu Activation
        return np.maximum(0, X)

    def bwd_propagation(self, X, dX):
        X = X > 0
        return dX * X

class Softmax:
    "Softmax output layer with cross entropy loss function"
    def __init__(self):
        self.logit = None
        self.y_exp = None

    def fwd_propagation(self, X, y):
        # Initialize weights for each sample
        self.y_exp = np.zeros(X.shape).reshape(-1)
        self.y_exp[y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1
        self.y_exp = self.y_exp.reshape(X.shape)
        # Cross Entropy
        delta = X - np.amax(X, axis=1, keepdims=True)
        delta_exp_sum = (np.sum(np.exp(delta), axis=1, keepdims=True))
        self.logit = np.exp(delta) / delta_exp_sum
        return -np.sum((delta - np.log(delta_exp_sum)) * self.y_exp) / X.shape[0]

    def bwd_propagation(self, X):
        return -(self.y_exp - self.logit) / X.shape[0]

def construct_model():
    """4 fully connected linear layers with Relu activation with 8 neurons in each layer"""
    model = dict()
    model['l1'] = Linear(2, 8)
    model['a1'] = Relu()
    model['l2'] = Linear(8, 8)
    model['a2'] = Relu()
    model['l3'] = Linear(8, 8)
    model['a3'] = Relu()
    model['l4'] = Linear(8, 2)
    model['loss'] = Softmax()
    return model

def update_parameters(model, learning_rate, X, y):
    """Update parameter using backward propagation and mini-batch gradient descent"""
    # Forward propagation
    l1_output = model['l1'].fwd_propagation(X)
    a1_output = model['a1'].fwd_propagation(l1_output)
    l2_output = model['l2'].fwd_propagation(a1_output)
    a2_output = model['a2'].fwd_propagation(l2_output)
    l3_output = model['l3'].fwd_propagation(a2_output)
    a3_output = model['a3'].fwd_propagation(l3_output)
    l4_output = model['l4'].fwd_propagation(a3_output)
    loss = model['loss'].fwd_propagation(l4_output, y)
    # Backward propagation
    l4_grad = model['loss'].bwd_propagation(l4_output)
    a3_grad = model['l4'].bwd_propagation(a3_output, l4_grad)
    l3_grad = model['a3'].bwd_propagation(l3_output, a3_grad)
    a2_grad = model['l3'].bwd_propagation(a2_output, l3_grad)
    l2_grad = model['a2'].bwd_propagation(l2_output, a2_grad)
    a1_grad = model['l2'].bwd_propagation(a1_output, l2_grad)
    l1_grad = model['a1'].bwd_propagation(l1_output, a1_grad)
    input_grad = model['l1'].bwd_propagation(X, l1_grad)
    # Mini-batch Gradient Descent
    for key, layer in model.items():
        if hasattr(layer, 'params'):
            layer.params['W'] -= layer.params['dW'] * learning_rate
            layer.params['b'] -= layer.params['db'] * learning_rate

    return loss

def make_predictions(X, model):
    """Make prediction by taking the output class with the largest probability"""
    l1_output = model['l1'].fwd_propagation(X)
    a1_output = model['a1'].fwd_propagation(l1_output)
    l2_output = model['l2'].fwd_propagation(a1_output)
    a2_output = model['a2'].fwd_propagation(l2_output)
    l3_output = model['l3'].fwd_propagation(a2_output)
    a3_output = model['a3'].fwd_propagation(l3_output)
    l4_output = model['l4'].fwd_propagation(a3_output)
    predictions = np.argmax(l4_output, axis=1).reshape(l4_output.shape[0])
    return predictions, l4_output

def calculate_model_performance(X, y, model):
    """Calculate loss and accuracy for performance benchmarking"""
    predictions, l4_output = make_predictions(X, model)
    loss = model['loss'].fwd_propagation(l4_output, y)
    accuracy = np.sum(predictions == y)/len(X)

    return loss, accuracy

def main():
    start_time = time.time()
    # Set seed for reproducibility
    np.random.seed(1)
    # Read training data
    X = np.genfromtxt(sys.argv[1], delimiter=',', skip_header = 0)
    y = np.genfromtxt(sys.argv[2], delimiter=',', skip_header = 0)
    # Create validation set
    X_train, X_val = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_val = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]
    # Read test input data
    X_test = np.genfromtxt(sys.argv[3], delimiter=',', skip_header = 0)

    # Construct model
    model = construct_model()
    best_model = copy.deepcopy(model)
    best_loss = np.inf
    patience = STOP
    epoch_number = 0
    # Train model
    while patience > 0:
        print('Epoch ', epoch_number)
        # Randomize training data for each epoch, sample without replacement
        total_loss = 0
        indexes = np.random.permutation(len(X_train))
        X_train_epoch, y_train_epoch = X_train[indexes], y_train[indexes]
        total_iterations = np.floor(len(X_train_epoch) / BATCH_SIZE)
        # Update model neuron weights
        for j in range(int(total_iterations)):
            X_train_epoch_batch = X_train_epoch[np.arange(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)]
            y_train_epoch_batch = y_train_epoch[np.arange(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)]
            loss = update_parameters(model, LEARN_RATE, X_train_epoch_batch, y_train_epoch_batch)
            total_loss += loss
        # Calculate train and validation losses
        train_loss, train_acc = calculate_model_performance(X_train_epoch, y_train_epoch, model)
        val_loss, val_acc = calculate_model_performance(X_val, y_val, model)
        # Cross validation with early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience = STOP
            # Save best model for prediction
            best_model = copy.deepcopy(model)
        else:
            patience -= 1

        print('Training Loss: ', train_loss, 'Training Accuracy: ', train_acc)
        print('Validation Loss: ', val_loss, 'Validation Accuracy: ', val_acc)

        epoch_number += 1

    end_time = time.time()

    print('Training time: ', end_time - start_time)
    # Create predictions based on best model
    test_predictions, _ = make_predictions(X_test, best_model)

    # Save prediction results to csv
    np.savetxt("test_predictions.csv", test_predictions.astype(int), fmt='%i', delimiter="")

if __name__ == "__main__":
    main()