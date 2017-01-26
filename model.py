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
    dataset = dataset.reshape((-1, image_width * image_height)).astype(np.float32)
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
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_width * image_height))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([image_width * image_height, num_hidden_nodes]))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    logits = tf.matmul(hidden, weights2) + biases2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden, weights2) + biases2)
    test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(
        tf.matmul(test_hidden, weights2) + biases2)

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
