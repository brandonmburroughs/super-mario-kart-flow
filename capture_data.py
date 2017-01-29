#!/usr/bin/env python

"""Capture images and inputs from controller."""

import os
import threading
from time import sleep

import inputs
from utils import mkdir, grab_screen


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

    def run(self):
        """Run the image writing process.

        Returns
        -------
        None
        """
        try:
            save_image(self.img, self.i, self.last_event)
        except:
            pass


def save_image(img, i, last_event):
    """Write an image to disk.

    Parameters
    ----------
    img : :obj:`Image`
        The PIL Image
    i : int
        The current image number
    last_event : :obj:`InputEvent`
        The most recent input event from the controller

    Returns
    -------
    None
    """
    mkdir("data")
    img.save("data/image_{}_{}_{}_{}.png".format(
        i,
        last_event.ev_type,
        last_event.code,
        last_event.state
    ))


def get_most_recent_image_number(data_folder):
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
        return 0
    image_numbers = [int(image.split("_")[1]) for image in images]
    return max(image_numbers)


def record():
    """Record the top left corner of the screen.

    Returns
    -------
    None
    """
    i = get_most_recent_image_number("data") + 1
    last_event = None
    while True:
        # Full screen
        #img = grab_screen(0, 52, 512, 396)
        # Top half for Super Mario kart
        img = grab_screen(0, 58, 512, 152)
        events = inputs.get_gamepad()
        for event in events:
            if event.ev_type == "Key" or event.ev_type == "Absolute":
                last_event = event
        ImageWriter(img, i, last_event).start()
        i += 1
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
    record()
