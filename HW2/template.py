import pickle
import numpy as np
import nltk
from utils import extract_features

nltk.download('twitter_samples')
nltk.download('stopwords')


# @title Q1.1 Sigmoid Inference

def sigmoid(x):
    """
    This function implements the logistic regression algorithm.

    Args:
      x: A numpy array of shape (d,) or (N, d) representing
        the input data.

    Returns:
      y: A numpy array of shape (l,) or (N, l) representing
        the output data.
    """
    return None


# @title Q1.2 One Layer Inference

def inference_layer(X, W, b):
    """
    Implements the forward propagation for the logistic regression model.

    Args:
      X: The input data, of shape (number of features, number of examples).
      w: Weights, a numpy array of shape (number of features, 1).
      b: Bias, a scalar.

    Returns:
      y: The output of shape l or l x N
    """
    return None


# @title Q1.3 Two Layer Inference

def inference_2layers(X, W1, w2, b1, b2):
    """
    Implements the forward propagation of a two layer neural network
    model.
  
    - Let d be the number of features (dimensionality) of the input.
    - Let N be the number of data samples in your batch
    - Let H be the number of hidden units
  
    Args:
    X: The input data, with shape either d or d x N
    W1: Weights for the first weight matrix: shape = H x d
    w2: Weights for the second matrix: shape = 1 x H
    b1: Bias value for first layer
    b2: Bias value for second layer
  
    Returns:
    y: Singular output data (1, 0), with shape either 1 or N.
    """
    return None


# @title Q1.4 BCE Loss

def bce_forward(yhat, y):
    """
    This function calculates the gradient of the logistic regression cost
    function with respect to the weights and biases.

    Args:
      yhat: A numpy array of shape (N,) representing the predicted outputs
      y: A numpy array of shape (N,) representing the target labels.

    Returns:
      loss_value
    """
    return None


# @title Q3.1 Gradients

def gradients(X, y, W1, W2, b1, b2):
    '''
    Calculate the gradients of the cost functions with respect to W1, W2, b1, and b2.

    Args:
      W1: Weight np.array of shape (h, d), first layer
      b1: Bias np.array of shape (h,), first layer
      W1: Weight np.array of shape (1, h), second layer
      b1: Bias value, second layer
      X: Input data of shape (d, N) representing the input data.
      y: Target / output labels of shape (N,)

    Returns:
      dL/dW1: Derivative of cost fxn w/r/t W1
      dL/db1: Derivative of cost fxn w/r/t b1
      dL/dW2: Derivative of cost fxn w/r/t W2
      dL/db2: Derivative of cost fxn w/r/t b2
      L: Loss value
    '''
    return None, None, None, None, None


# @title Q4 Parameter Updates

def update_params(batchx, batchy, W1, B1, W2, b2, lr=0.01):
    '''
    Updates your parameters
        batchx: Mini-batch of features
        batchy: Corresponding mini-batch of labels
        W1: Starting value of W1
        b1: Starting value of b1
        W2: Starting value of W2
        b2: Starting value of b2
        lr: Learning parameter, default to 0.01

    Returns
        W1: New value of W1
        b1: New value of b1
        W2: New value of W2
        b2: New value of b2
    '''
    return None, None, None, None


# @title Q5 Training the Neural Network

def train_nn(filename, hidden_layer_size, iters=1e6, lr=0.01):
    '''
    Reads in data (twitter_data.pkl), initiates parameters, extracts features
    and returns the parameters

    Returns:
        W1, b1, W2, b2
    '''
    return None, None, None, None