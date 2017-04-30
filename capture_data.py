#!/usr/bin/env python

"""Capture images and inputs from controller."""

import os
import threading
from time import sleep

import inputs
from utils import mkdir, grab_screen


class Exception_TIMEOUT(Exception):
    """Special timeout exception."""
    pass


class Timeout:
    """Timeout class."""
    def __init__(self, f, seconds=1.0, error_message='Timeout'):
        self.seconds = seconds
        self.thread = threading.Thread(target=f)
        self.thread.daemon = True
        self.error_message = error_message

    def handle_timeout(self):
        raise Exception_TIMEOUT(self.error_message)

    def __enter__(self):
        try:
            self.thread.start()
            self.thread.join(self.seconds)
        except Exception, te:
            raise te

    def __exit__(self, type, value, traceback):
        if self.thread.is_alive():
            return self.handle_timeout()


class ImageWriter(threading.Thread):
    """Asynchronous image writer."""

    def __init__(self, img, i, last_event):
        """Initialize an ImageWriter.

        Parameters
        ----------
        img : :obj:`Image`
            The PIL Image
        i : int
            The current image number
        last_event : :obj:`InputEvent`
            The most recent input event from the controller
        """
        threading.Thread.__init__(self)
        self.img = img
        self.i = i
        self.last_event = last_event

    def save_image(self):
        """Write an image to disk."""
        self.img.save("data/image_{}_{}_{}_{}.png".format(
            self.i,
            self.last_event.ev_type,
            self.last_event.code,
            self.last_event.state
        ))

    def run(self):
        """Run the image writing process."""
        try:
            self.save_image()
        except:
            pass


class Recorder(object):
    """Recorder object."""

    def __init__(self):
        self.last_event = None
        self.image_number = None

    def get_most_recent_image_number(self, data_folder):
        """Given a directory of images, get the most recent image number.

        Parameters
        ----------
        data_folder : str
            The folder containing the images

        Returns
        -------
        int
            The most recent image number
        """
        data_folder = os.path.join(os.getcwd(), data_folder)
        images = os.listdir(data_folder)
        if len(images) == 0:
            self.image_number = 0
        else:
            image_numbers = [int(image.split("_")[1]) for image in images]
            self.image_number = max(image_numbers) + 1

    def get_event(self):
        """Get the most recent button push."""
        events = inputs.get_gamepad()
        for event in events:
            if event.ev_type == "Key" or event.ev_type == "Absolute":
                self.last_event = event

    def record(self):
        """Record the top left corner of the screen.

        Returns
        -------
        None
        """
        mkdir("data")
        self.get_most_recent_image_number("data")
        while True:
            # Full screen #img = grab_screen(0, 52, 512, 396)
            # Top half for Super Mario kart
            try:
                with Timeout(self.get_event, seconds=0.1):
                    pass
            except Exception_TIMEOUT:
                pass
            img = grab_screen(0, 58, 512, 152)
            ImageWriter(img, self.image_number, self.last_event).start()
            self.image_number += 1
            sleep(0.1)


def print_output():
    """Print the controller output.

    Returns
    -------
    None
    """
    events = inputs.get_gamepad()
    for event in events:
        print(event.ev_type, event.code, event.state)

if __name__ == "__main__":
    Recorder().record()
