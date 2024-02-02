# CSE493G
Deep Learning 
# KNN Classifier

- **Training** is easy and fast, predicting is slow, need to calculate distance to every image
- **Distance metrics**

## Predict:

- For each test image:
  - Find the closest training images, predict the label of the nearest images
  - For each test image, train O(1), predict O(N)
- k=1 v. k=n
  - **k=1:**
    - not robust to label noise
    - Overfitting on training images, training accuracy = 100%
  - **k=n**
    - Take a majority vote from K closest points
    - ↑ generalization
    - Smooth out decision boundaries
    - ↑ white areas - uncertain - ethical errors?

## Setting hyperparameters

- Should be on the validation set
  - On train set: bad because k=1 would overfit on the training set
  - On test set: bad because no idea how the algorithm will perform on new data

## Curse of dimensionality

# Linear Classifier

- Hard cases: If linear separable?
- Bias: if we don’t see X, what you’re most likely looking at

## Loss functions = regret

- SVM loss = hinge loss
  - 0 <= L_i < infinity

## Regularization

- Why?
  - Express preferences over weights -> 使L更小
  - Make the model simple so it works on test data
  - Improve optimization by adding curvature
- **Lamda:**
  - Big lamda value pick the simplest model
  - Small lamda care about data loss more, more fit
- **L2:** likes to spread out weights
- **L1:** prefers sparsity, i.e. lots of 0s

## Softmax - cross entropy - probabilities

- Min: -log(1) = 0, when the class is correct
- Max: infinity, -log(0)

## Optimization

- **Numerical gradient:** approximate, slow, easy to write
- **Analytic gradient:** exact, fast, error-prone
- **Gradient:** Gradient tells you where is a direction changing the most; Not necessarily meaning if keep going in that direction, you’ll hit the lowest point
- **SGD:** Stochastic Gradient Descent
  - Batch size ↑, avoid bias

# Neural Network

- Fully-connected networks, MLP
  - ↑ neurons, decision boundaries are more complicated

## Backpropagation: