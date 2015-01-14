# Marmot

Neural network framework built on Theano. Marmot was designed to make it really easy to implement custom architectures. It's optimized for fast training on GPUs.

Features:

- Feedforward and recurrent nets
- SGD with various learning rate strategies
  - Currently only fixed LR and Adadelta, but easy to add others
- CTC (Connectionist Temporal Classification) for training RNNs without prior sequence alignment
- L2 regularization, early stopping

Coming soon:

- Dropout, rectified linear units
- Support for large datasets that don't fit in memory
