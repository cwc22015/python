from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:f = np.zeros(num_tests)

    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    #  Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    probs = 0.0
    probs_der = 0.0
    f_max = 0.0
    num_tests = X.shape[0]
    f = np.zeros(num_tests)
    for i in range(num_tests):
        f[i] = X[i].dot(W)
        probs += np.exp(X[i].dot(W))
        probs_der += np.transpose(X[i]) * np.exp(X[i].dot(W))
        if f[i] > f_max:
            f_max = f[i]
        
    # make numeric stable
    f = f - f_max
    probs /= np.exp(f_max)
    probs_der /= np.exp(f_max)
    for i in range(num_tests):
        loss += np.exp(f[y[i]]) / probs
    #pass

    # calculate dW
    for i in range(num_tests):
        dW += np.transpose(X[y[i]]) * np.exp(X[y[i]].dot(W)) / probs - (np.exp(X[y[i]].dot(W)) * probs_der / probs / probs)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
