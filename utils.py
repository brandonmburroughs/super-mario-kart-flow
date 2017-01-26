"""
Utilities
"""

import os


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
