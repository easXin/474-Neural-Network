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

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #creating a column for the bias terms and inserting at the end
    N = train_data.shape[0]
    hidden_bias = np.ones(N)
    train_data = np.column_stack((train_data, hidden_bias))

    #foward pass to hidden layer
    hiddenlayer_output = sigmoid(np.matmul(train_data, w1.T))

    #same process as above excpet its from hidden -> output
    output_bias = np.ones(hiddenlayer_output.shape[0])
    hiddenlayer_output = np.column_stack((hiddenlayer_output, output_bias))
    outputlayer_output = sigmoid(np.matmul(hiddenlayer_output, w2.T))

    #stores the encoding of the output
    outclass = np.zeros(np.shape(outputlayer_output))
    i = 0

    #computes the 1 of k encoding
    while i < len(outclass):
        for j in range(np.shape(outclass)[1]):
            if j == int(train_label[i]):
                outclass[i][j] = 1
        i += 1
    # compute the error function
    obj_val = (-1 / len(train_data)) * np.sum(np.add(outclass * np.log(outputlayer_output), (1 - outclass) * np.log(1 - outputlayer_output)))

    #regularize the data using equation 15
    outside_constant = lambdaval / (2 * N)
    first_summation = np.sum(np.square(w1), axis=1)
    second_summation = np.sum(np.square(w2), axis=1)
    result = np.sum(first_summation,axis=0) + np.sum(second_summation, axis=0)
    obj_val = obj_val + (outside_constant * result)

    # Calculate Gradient
    delta = np.subtract(outputlayer_output,outclass)
    delta_transpose = delta.transpose()

    #initializing the gradient weights
    gradient1 = np.zeros(w1.shape)
    gradient2 = np.zeros(w2.shape)


    #calculate gradient for hidden -> output weights
    gradient2 = (1/N) * np.dot(delta_transpose, hiddenlayer_output)
    constant_term = (lambdaval * w2) / N
    gradient2 = gradient2 + constant_term

    #calculate the gradient for input -> hidden weights

    multiplying_term = np.subtract(1,hiddenlayer_output)
    multiplying_term = np.multiply(multiplying_term,hiddenlayer_output)
    hidden_delta = np.dot(delta,w2)
    multiplying_term = multiplying_term * hidden_delta
    multiplying_term_transpose = multiplying_term.transpose()

    gradient1 = (1/N) * np.dot(multiplying_term_transpose,train_data)
    constant_term = (lambdaval * w1) / N

    gradient1= np.delete(gradient1,n_hidden,0)
    gradient1 = gradient1 + constant_term

    obj_grad = np.concatenate((gradient1.flatten(), gradient2.flatten()), 0)
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

    vectOfOnes = np.ones(len(data))  # vertical vector of ones with same number of rows as data
    data = np.column_stack((data, vectOfOnes))  # combines data with the vector; vector is now a column of data

    dotProduct1 = np.dot(data, W1.T)  # W * x;messed up before didnt need to transpose
    outputLayer1 = sigmoid(dotProduct1)  # computes sigmoid of each output

    # now for W2; basically pretty much the same thing
    vectOfOnes = np.ones(len(outputLayer1))  # vertical vector of ones with same number of rows as the first output layer
    outputLayer1 = np.column_stack((outputLayer1, vectOfOnes))

    dotProduct2 = np.dot(outputLayer1,W2.T)  # W * x
    outputLayer2 = sigmoid(dotProduct2)

    labels = np.argmax(outputLayer2, axis=1)

    return labels

