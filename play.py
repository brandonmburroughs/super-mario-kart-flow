"""
Play the game!
"""

import sys
import numpy as np
import tensorflow as tf

from evdev import UInput, ecodes as e
from time import sleep

from utils import grab_screen

image_width = 512
image_height = 210
pixel_depth = 255.0
batch_size = 128
num_hidden_nodes = 1024
num_labels = 5

key_map = {
    'Absolute_ABS_HAT0X_-1.png': "left",
    'Absolute_ABS_HAT0X_0.png': "center",
    'Absolute_ABS_HAT0X_1.png': "right",
    'Key_BTN_SOUTH_0.png': "stop_going",
    'Key_BTN_SOUTH_1.png': "go"
}


class Predictor(object):
    """Make prediction with a trained tf model."""
    def __init__(self, trained_model_path):
        """Initialize a predictor.

        Parameters
        ----------
        trained_model_path: File path to the trained model.
        """
        self.trained_model_path = trained_model_path
        self.sess = None
        self.weights1 = None
        self.biases1 = None
        self.weights2 = None
        self.biases2 = None
        self.label_names = np.array([
            'Absolute_ABS_HAT0X_-1.png',
            'Absolute_ABS_HAT0X_0.png',
            'Absolute_ABS_HAT0X_1.png',
            'Key_BTN_SOUTH_0.png',
            'Key_BTN_SOUTH_1.png'
        ])

    def start_session(self):
        """Start a tensorflow session and load the model."""
        self.sess = tf.InteractiveSession()

        self.weights1 = tf.Variable(tf.truncated_normal([image_width * image_height, num_hidden_nodes]), name="weights_1")
        self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]), name="biases_1")
        self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]), name="weights_2")
        self.biases2 = tf.Variable(tf.zeros([num_labels]), name="biases_2")

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.trained_model_path)

    @staticmethod
    def prepare_image(image):
        """Prepare the image for prediction

        Parameters
        ----------
        image : A PIL image

        Returns
        -------
        numpy array
            A numpy array of the scaled image
        """
        scaled_image = (np.array(image).mean(axis=2).astype(float) - pixel_depth / 2) / pixel_depth
        return scaled_image.reshape((-1, image_width * image_height)).astype(np.float32)

    def predict(self, image):
        """Make a prediction given an image.

        Parameters
        ----------
        image : A numpy array image

        Returns
        -------
            A predicted controller label name
        """
        tf_test_dataset = self.prepare_image(image)
        test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, self.weights1) + self.biases1)
        test_prediction = tf.nn.softmax(
            tf.matmul(test_hidden, self.weights2) + self.biases2)

        predictions = test_prediction.eval()
        return self.label_names[predictions.argmax()]


class KeyManager(object):
    """Key manager for driving."""
    def __init__(self):
        self.current_direction = "center"
        self.stop_go = "go"
        self.ui = UInput()
        self.key_mapping = {
            "left": e.KEY_LEFT,
            "right": e.KEY_RIGHT,
            "go": e.KEY_B
        }
        self.ui.write(e.EV_KEY, self.key_mapping["go"], 1)
        self.ui.syn()

    def press_key(self, direction):
        """Press a key.

        Parameters
        ----------
        direction : str
            The direction corresponding to the key to be pressed

        Returns
        -------
        None
        """
        if direction == "left" and (self.current_direction == "center" or self.current_direction == "right"):
            key = self.key_mapping[direction]
            self.ui.write(e.EV_KEY, key, 1)
            self.ui.syn()
            self.current_direction = "left"
        elif direction == "right" and (self.current_direction == "center" or self.current_direction == "left"):
            key = self.key_mapping[direction]
            self.ui.write(e.EV_KEY, key, 1)
            self.ui.syn()
            self.current_direction = "right"
        elif direction == "center" and self.current_direction != "center":
            key = self.key_mapping[self.current_direction]
            self.ui.write(e.EV_KEY, key, 0)
            self.ui.syn()
            self.current_direction = "center"
        elif direction == "go":
            if self.stop_go == "stop":
                key = self.key_mapping[direction]
                self.ui.write(e.EV_KEY, key, 1)
                self.ui.syn()
                self.stop_go = "go"
        elif direction == "stop_going":
            if self.stop_go == "go":
                key = self.key_mapping[self.stop_go]
                self.ui.write(e.EV_KEY, key, 0)
                self.ui.syn()
                #sleep(0.1)
                #self.ui.write(e.EV_KEY, key, 1)
                #self.ui.syn()
                self.stop_go = "stop"

        sys.stdout.write("Direction: {}    Stop/Go: {}\r".format(self.current_direction[0], self.stop_go[0]))
        sys.stdout.flush()
        sleep(0.1)

def play(trained_model_path):
    """Play Super Mario Kart.ert

    Parameters
    ----------
    trained_model_path : str
        The path for the trained tf model

    Returns
    -------
    None
    """
    predictor = Predictor(trained_model_path)
    predictor.start_session()

    key_manager = KeyManager()

    while True:
        screen_grab = grab_screen(0, 58, 512, 152)
        prediction = key_map[predictor.predict(screen_grab)]
        key_manager.press_key(prediction)


if __name__ == "__main__":
    play("model/super-mario-kart-flow")
