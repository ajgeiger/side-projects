# Convolutional Neural Network.
# Implemented linear classifiers to recognize handwritten digits from images.
# the dataset utilized the MNIST dataset. The multiclass solution involved
# using a deep convolution network which worked by applying each new image all
# pairwise classifiers and performing a majority vote to decide the output digit.
# k-fold Cross-validation was used to evaluate model performance. The calculated
# performance metrics of the different folds were combined and aggregated
# to a data structure. Note the original program was done in Matlab. The current
# program before you has been reworked using Tensor Flow
# (http://yann.lecun.com/exdb/mnist/)
#
# Author: Alex Geiger
# email: ajg1444@rit.edu


# Import MNIST dataset and tensorflow objects:
from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Training Parameters
learning_rate = 0.001
num_steps  = 2000
batch_sz = 128

# Network Parameters
pixels_x = 28    # number of pixels along the x-axis
pixels_y = 28    # number of pixels along the y-axis
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25   # Dropout probability


# compact settings into a dictionary to simplify passing data between functions
net_settings = {"num_classes": num_classes,"dropout":dropout,"pixels_x": pixels_x,"pixels_y": pixels_y}


# Create the neural network
def build_network(features, net_settings, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Data Preprocessing: MNIST data input is a 1x784 vector representing a 28x28 pixels
        images_features = features['images'] # obtain images from the
        images_features = tf.reshape(images_features,shape=[-1,net_settings['pixels_x'],net_settings['pixels_y'],1])

        # (32 filters & kernel size of 5 Convolution Layer) && (max pooling strides & kernel both size 2)
        layer_1 = tf.layers.conv2d(images_features, 32, 5, activation=tf.nn.relu)
        layer_1 = tf.layers.max_pooling2d(layer_1, 2, 2)

        # (64 filters & kernel size of 3 Convolution Layer) && (max pooling strides & kernel both size 2)
        layer_2 = tf.layers.conv2d(layer_1, 64, 3, activation=tf.nn.relu)
        layer_2 = tf.layers.max_pooling2d(layer_2, 2, 2)

        # tune network:
        conv_net = tf.contrib.layers.flatten(layer_2) # Create a 1-D input while retaining the batch size
        conv_net = tf.layers.dense(conv_net, 1024)    # set Integer or Long, dimensionality of the output space
        conv_net = tf.layers.dropout(conv_net, rate=net_settings['dropout'], training=is_training) # Apply Dropout
    return tf.layers.dense(conv_net,net_settings['num_classes']) # set number of classes in multiclass



# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):

    # Train Test and Predict Network:
    train_network = build_network(features,net_settings, reuse=False, is_training=True)
    test_network = build_network(features,net_settings, reuse=True, is_training=False)
    pred_classes = tf.argmax(test_network, axis=1)


    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_optimizer = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = train_network, labels = tf.cast(labels, dtype=tf.int32)))

    # Define network optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_optimizer = optimizer.minimize(loss_optimizer, global_step=tf.train.get_global_step())
    accuracy_optimizer = tf.metrics.accuracy(labels=labels, predictions=pred_classes) # Evaluate model accuracy

    # TF Estimators requires to return a EstimatorSpec, that specify
    estimation = tf.estimator.EstimatorSpec(mode=mode,predictions=pred_classes,loss=loss_optimizer,
                                            train_op=train_optimizer,
                                            eval_metric_ops={'accuracy': accuracy_optimizer})

    return estimation



# instantiate the Estimator
model = tf.estimator.Estimator(model_fn)
y_lable = mnist.train.labels
images = {'images': mnist.train.images}


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(x=images,y=y_lable,batch_size=batch_sz, num_epochs=None,shuffle=True)
model.train(input_fn, steps=num_steps) # Train the Model then evaluate model performance
input_fn = tf.estimator.inputs.numpy_input_fn(x=images,y=mnist.test.labels,batch_size=batch_sz,shuffle=False)
evaluation = model.evaluate(input_fn) # Use the Estimator 'evaluate' method


# print findings:
print("Accuracy:    ", evaluation['accuracy'])
print("Loss:        ", evaluation['loss'])
print("Global Step: ", evaluation['global_step'])
