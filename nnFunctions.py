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
        s = 1 / (1 + exp(-1 * z))
    else:
        s = np.zeros(z.shape)
        for ix, iy in np.ndindex(z.shape):
            temp = z[ix,iy]
            sigmoid_result = 1 / (1 + exp(-1 * temp))
            print(str(sigmoid_result))
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
	training_data = np.column_stack((train_data, np.ones(train_data.shape[0])))
    tempOldDa = sigmoid(np.dot(np.column_stack((sigmoid(np.dot(training_data, W1.T)), np.ones(sigmoid(np.dot(training_data, W1.T)).shape[0]))), W2.T))
   	t1 = tempOldDa - train_label
    g2 = np.dot(t1.T, np.column_stack((sigmoid(np.dot(training_data, W1.T)), np.ones(sigmoid(np.dot(training_data, W1.T)).shape[0]))))
    g1 = np.dot(((1 - np.column_stack((sigmoid(np.dot(training_data, W1.T)), np.ones(sigmoid(np.dot(training_data, W1.T)).shape[0])))) * np.column_stack((sigmoid(np.dot(training_data, W1.T)), np.ones(sigmoid(np.dot(training_data, W1.T)).shape[0]))) * (np.dot(delta_l, W2))).T, training_data)
    g1 = np.delete(g1, n_hidden, 0)
    obj_grad = np.array([]).concatenate((g1.flatten(), g2.flatten()), 0) / training_data.shape[0] 
    obj_val= (np.sum(-1 * (train_label * np.log(tempOldDa) + (1 - train_label) * np.log(1 - tempOldDa))) / training_data.shape[0])+((lambdaval / (2 * training_data.shape[0])) * (np.sum(np.square(W1)) + np.sum(np.square(W2))))
    return (obj_val, np.zeros(t0))

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
