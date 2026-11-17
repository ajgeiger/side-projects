# MNIST digit classification

A convolutional neural network that classifies handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). I first wrote this in MATLAB and later rewrote it in Python with TensorFlow.

## How it works

`cnn_mnist.py` builds the network with the TensorFlow Estimator API:

- Two convolution layers (32 filters with a 5x5 kernel, then 64 filters with a 3x3 kernel), each followed by 2x2 max pooling
- A dense layer with 1024 units and 25% dropout
- An output layer for the 10 digit classes

It trains for 2000 steps with the Adam optimizer (learning rate 0.001, batch size 128), then prints accuracy, loss, and the global step.

## Running it

The script uses TensorFlow 1.x APIs (`tf.layers`, `tf.contrib`, and the old MNIST tutorial loader), so it needs an older TensorFlow 1 install to run as is.

```
python cnn_mnist.py
```

The data downloads to `/tmp/data/` the first time it runs.
