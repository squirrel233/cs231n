import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_samples = y.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  Scores = X.dot(W)
  cons = np.max(Scores, axis=1)  #for numeric stability
  Scores = (Scores.T - cons).T

  for i in range(num_samples):
      exp_sum = np.sum(np.exp(Scores[i,:]))
      loss += np.log(exp_sum) - Scores[i,y[i]]
      dW[:,y[i]] -= X[i,:]
      for j in range(num_classes):
          dW[:,j] += (np.exp(Scores[i,j])/exp_sum) * X[i,:].T

  loss /= num_samples                   # order can't
  loss += 0.5 * reg * np.sum(W*W)       # be wrong
  dW /= num_samples
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_samples = y.shape[0]

  Scores = X.dot(W)
  cons = np.max(Scores, axis=1)  #for numeric stability
  Scores = (Scores.T - cons).T
  exp_sum = np.sum(np.exp(Scores), axis=1)
  loss -= np.sum(Scores[np.array(range(num_samples)),y])
  loss += np.sum(np.log(exp_sum))
  loss /= num_samples
  loss += 0.5 * reg * np.sum(W*W)

  Logic = np.zeros_like(Scores)   #(num, 10classes)
  Logic[np.array(range(num_samples)),y] = -1
  Logic += (np.exp(Scores.T)/exp_sum).T
  dW += X.T.dot(Logic)
  dW /= num_samples
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
