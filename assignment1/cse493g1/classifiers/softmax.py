from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        s_j_exp = 0.0

        for j in range(num_classes):
            s_j_exp += np.exp(scores[j])

        for j in range(num_classes):
            softmax_prob = np.exp(scores[j]) / s_j_exp
            if j == y[i]:
                # S_yi = 1
                dW[:, j] += (softmax_prob - 1) * X[i]
            else:
                dW[:, j] += softmax_prob * X[i]

        loss += -np.log(np.exp(correct_class_score) / s_j_exp)
        
    # regularization
    loss /= num_train
    dW /= num_train 
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

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

    num_train = X.shape[0]  # N
    scores = X.dot(W)  # N x C
    
    s_j_exp = np.sum(np.exp(scores), axis=1)  # N x 1 sum for each row
    softmax_probs = np.exp(scores) / s_j_exp[:, np.newaxis]  # N x C
    correct_class_probs = softmax_probs[np.arange(num_train), y]  # N x 1
    loss = np.sum(-np.log(correct_class_probs))

    softmax_probs[np.arange(num_train), y] -= 1  # S_yi = 1
    dW = X.T.dot(softmax_probs)

    loss /= num_train
    dW /= num_train

    # regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W  # D x C

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
