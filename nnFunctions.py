import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    #if its a scalar then just do sigmoid activation
    #else iterate through the matrix and do sigmoid activation on each element and store them in s
    if np.isscalar(z):
        s = 1 / (1 + np.exp(-1 * z))
    else:
        s = np.zeros(z.shape)
        for ix, iy in np.ndindex(z.shape):
            temp = z[ix,iy]
            sigmoid_result = 1 / (1 + np.exp(-1 * temp))
            s[ix,iy] = sigmoid_result
    return s

def nnObjFunction(params, *args):
    '''
        % nnObjFunction computes the value of objective function (cross-entropy
        % with regularization) given the weights and the training data and lambda
        % - regularization hyper-parameter.

        % Input:
        % params: vector of weights of 2 matrices W1 (weights of connections from
        %     input layer to hidden layer) and W2 (weights of connections from
        %     hidden layer to output layer) where all of the weights are contained
        %     in a single vector.
        % n_input: number of node in input layer (not including the bias node)
        % n_hidden: number of node in hidden layer (not including the bias node)
        % n_class: number of node in output layer (number of classes in
        %     classification problem
        % train_data: matrix of training data. Each row of this matrix
        %     represents the feature vector of a particular image
        % train_label: the vector of truth label of training images. Each entry
        %     in the vector represents the truth label of its corresponding image.
        % lambda: regularization hyper-parameter. This value is used for fixing the
        %     overfitting problem.

        % Output:
        % obj_val: a scalar value representing value of error function
        % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
        % NOTE: how to compute obj_grad
        % Use backpropagation algorithm to compute the gradient of error function
        % for each weights in weight matrices.
        '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # insert bias into training data
    train_data = np.insert(train_data, 0, 1, axis=1)

    # number of inputs that the network will compute outputs to
    N = train_data.shape[0]

    # stores the outpult result for each output node
    output_result = np.zeros(n_class)

    # outerloop is what will do foward pass and back propogation
    for i in range(N):

        # is the current input into the network
        xi = train_data[i]

        # stores the delta for each output unit
        deltas_output = np.zeros(n_class)

        # stores the delta for each hidden unit
        deltas_hidden = np.zeros(n_hidden)

        # stores the result of the hidden layer at each node
        z_hidden_result = np.zeros(n_hidden)

        # computes foward pass from input to hidden
        for j in range(n_hidden):
            # the weights for the given hidden node
            W1_i = W1[j]

            # computes the dot product of the input and the weights for hidden node
            W1_mul_xi = np.dot(W1_i, xi)

            # computes the sigmoid activation for the hidden node.
            sigmoid_activation = sigmoid(W1_mul_xi)

            # stores the result of the sigmoid activation to its corresponding hidden node unit
            z_hidden_result[j] = sigmoid_activation

        # inserts the bias at the hidden layer
        z_hidden_result = np.insert(z_hidden_result, 0, 1)

        # starts the foward propogation from hidden layer to output layer
        # same algorithim as above
        for j in range(n_class):
            W2_i = W2[j]
            W2_mul_hiddeni = np.dot(W2_i, z_hidden_result)

            sigmoid_activation = sigmoid(W2_mul_hiddeni)

            output_result[j] = sigmoid_activation

        # computes the loss function of the outputs to the true label
        # loss = np.subtract(train_label, output_result) ** 2
        # loss = np.sum(loss) / 2

        # compute delta for the output
        for i in range(n_class):
            output_i = output_result[i]
            truelabel_yi = train_label[i]
            delta = output_i * (1 - output_i) * (truelabel_yi - output_i)
            deltas_output[i] = delta

    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    obj_grad = np.zeros(params.shape)

    return (obj_val, obj_grad)


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''

    labels = np.zeros((data.shape[0],))
    # Your code here

    return labels
