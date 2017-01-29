#!/usr/bin/env python

"""
Modeling.
"""

import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from utils import mkdir


image_width = 512
image_height = 210
pixel_depth = 255.0  # Number of levels per pixel.


def load_images(folder):
    """Load the images with data.

    Parameters
    ----------
    folder : str
        The folder containing the images

    Returns
    -------
    ndarray, ndarray
        The images and corresponding labels arrays
    """
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_height, image_width),
                         dtype=np.float32)
    labels = np.ndarray(shape=(len(image_files)), dtype=object)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).mean(axis=2).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_height, image_width):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            labels[num_images] = image.split("_", 2)[-1]
            num_images += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset, labels


def reformat(dataset, labels):
    """Reformat the width by height data into long vectors.

    Parameters
    ----------
    dataset : ndarray
        The ndarray with images
    labels : ndarray
        The ndarray with image labels

    Returns
    -------
    ndarray, ndarray
        The reformatted images and corresponding labels arrays
    """
    dataset = dataset.reshape((-1, image_height, image_width, 1)).astype(np.float32)
    label_names = np.array([
        'Absolute_ABS_HAT0X_-1.png',
        'Absolute_ABS_HAT0X_0.png',
        'Absolute_ABS_HAT0X_1.png',
        'Key_BTN_SOUTH_0.png',
        'Key_BTN_SOUTH_1.png'
    ])
    labels = (label_names == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    """Calculate the accuracy of predictions.

    Parameters
    ----------
    predictions : ndarray
        The predicted classes
    labels : ndarray
        The actual classes

    Returns
    -------
    float
        Percent of predictions that are correct
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])


def convolution_layer_variable_constructor(kernel_list):
    """Build convolution layer variables given a list kernels.

    Parameters
    ----------
    kernel_list : list
        A list of number of nodes in each layer

    Returns
    -------
    list (weights)
        The list of weight variables

    list (biases)
        The list of bias variables
    """
    weights = []
    biases = []
    for kernel_num in range(len(kernel_list)):
        weights.append(
            tf.Variable(
                tf.truncated_normal(
                    kernel_list[kernel_num]
                ), name="conv_weights_{}".format(kernel_num)
            )
        )

        biases.append(
            tf.Variable(
                tf.zeros(
                    [kernel_list[kernel_num][-1]]
                ), name="conv_biases_{}".format(kernel_num)
            )
        )

    return weights, biases


def convolution_layer_constructor(dataset, weights_list, biases_list, strides_list):
    """Given an original dataset, a list of weights, and a list of biases, construct the convolution layers.

    Parameters
    ----------
    dataset : tf dataset
        The original dataset

    weights_list : list of tf kernel weights Variables
        A list of kernel weights variables

    biases_list : list of tf bias Variables
        A list of bias variables

    strides_list : list
        A list of strides for each step

    Returns
    -------
    logit
        The logits for the output layer
    """
    conv_layers = [dataset]
    for i in range(len(weights_list)):
        conv_layers.append(
                tf.nn.relu(
                    tf.nn.conv2d(conv_layers[-1],
                                 weights_list[i],
                                 strides=[1, strides_list[i], strides_list[i], 1],
                                 padding="VALID"
                                 ) + biases_list[i]
                )
        )

    return conv_layers[-1]


def fully_connected_layer_variable_constructor(layer_list):
    """Build fully connected layer variables given a list of number of nodes in each layer.  The first layer should be the
    number of inputs while the last layer will be the number of outputs.

    Parameters
    ----------
    layer_list : list
        A list of number of nodes in each layer

    Returns
    -------
    list (weights)
        The list of weight variables

    list (biases)
        The list of bias variables
    """
    weights = []
    biases = []
    for layer_num in range(1, len(layer_list)):
        weights.append(
            tf.Variable(
                tf.truncated_normal(
                    [layer_list[layer_num - 1], layer_list[layer_num]]
                ), name="weights_{}".format(layer_num)
            )
        )

        biases.append(
            tf.Variable(
                tf.zeros(
                    [layer_list[layer_num]]
                ), name="biases_{}".format(layer_num)
            )
        )

    return weights, biases


def fully_connected_layer_constructor(dataset, weights_list, biases_list, keep_prob=1.0):
    """Given an original dataset, a list of weights, and a list of biases, construct the fully connected layers.

    Parameters
    ----------
    dataset : tf dataset
        The original dataset

    weights_list : list of tf weights Variables
        A list of weights variables

    biases_list : list of tf bias Variables
        A list of bias variables

    keep_prob : float
        The probability of keeping a layer (as opposed to dropping it out)

    Returns
    -------
    logit
        The logits for the output layer
    """
    layers = [dataset]
    for i in range(len(weights_list) - 1):
        layers.append(
            tf.nn.dropout(
                tf.nn.relu(
                    tf.matmul(layers[-1], weights_list[i]) + biases_list[i]
                ), keep_prob
            )
        )

    return tf.matmul(layers[-1], weights_list[-1]) + biases_list[-1]


# Load images
images, labels = load_images("data")
num_images = images.shape[0]
num_labels = len(set(labels))
indices = np.random.permutation(num_images)
training_idx = indices[:int(num_images * 0.7)]
valid_idx = indices[int(num_images * 0.7):int(num_images * 0.8)]
test_idx = indices[int(num_images * 0.8):]
train_dataset, train_labels = reformat(images[training_idx, :, :], labels[training_idx])
valid_dataset, valid_labels = reformat(images[valid_idx, :, :], labels[valid_idx])
test_dataset, test_labels = reformat(images[test_idx, :, :], labels[test_idx])


# Tensorflow graph
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Convolution kernels
    conv_kernel_list = [
        [5, 5, 1, 24],
        [5, 5, 24, 36],
        [5, 5, 36, 48],
        [3, 3, 48, 64],
        [3, 3, 64, 64]
    ]
    conv_stride_list = [2, 2, 2, 1, 1]
    conv_weights, conv_biases = convolution_layer_variable_constructor(conv_kernel_list)

    convs_output = convolution_layer_constructor(tf_train_dataset, conv_weights, conv_biases, conv_stride_list)
    convs_output_elements = convs_output.get_shape().num_elements() / batch_size

    # Fully connected layer weights and biases
    fcl_layer_list = [convs_output_elements, 1024, 256, 64, 16, num_labels]
    fcl_weights, fcl_biases = fully_connected_layer_variable_constructor(fcl_layer_list)

    # Training computation
    logits = fully_connected_layer_constructor(tf.reshape(convs_output, shape=(batch_size, convs_output_elements)),
                                               fcl_weights, fcl_biases)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data
    train_prediction = tf.nn.softmax(logits)
    valid_convs_output = convolution_layer_constructor(tf_valid_dataset, conv_weights, conv_biases, conv_stride_list)
    valid_logits = fully_connected_layer_constructor(tf.reshape(valid_convs_output,
                                                                shape=(valid_idx.shape[0], convs_output_elements)),
                                                     fcl_weights, fcl_biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_convs_output = convolution_layer_constructor(tf_test_dataset, conv_weights, conv_biases, conv_stride_list)
    test_logits = fully_connected_layer_constructor(tf.reshape(test_convs_output,
                                                               shape=(test_idx.shape[0], convs_output_elements)),
                                                    fcl_weights, fcl_biases)
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    saver = tf.train.Saver()
    mkdir("model")
    saver.save(session, "model/super-mario-kart-flow")
