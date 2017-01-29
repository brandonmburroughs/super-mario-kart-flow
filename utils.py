"""
Utilities
"""

import os
import ctypes

from PIL import Image



def mkdir(path):
    """
    Make a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        The directory to make

    Returns
    -------
    None
    """
    full_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)


# Load ctype function
LibName = 'prtscn.so'
AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
grab = ctypes.CDLL(AbsLibPath)


def grab_screen(x1, y1, x2, y2):
    """Grab a screen shot.

    Parameters
    ----------
    x1 : int
        x1 coordinate of rectangle capture window
    y1 : int
        y1 coordinate of rectangle capture window
    x2 : int
        x2 coordinate of rectangle capture window
    y2 : int
        y2 coordinate of rectangle capture window

    Returns
    -------
    Image
        The screenshot image
    """
    w, h = x1 + x2, y1 + y2
    size = w * h
    objlength = size * 3

    grab.getScreen.argtypes = []
    result = (ctypes.c_ubyte*objlength)()

    grab.getScreen(x1, y1, w, h, result)
    return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)
