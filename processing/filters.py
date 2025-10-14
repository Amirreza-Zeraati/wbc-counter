import cv2
import numpy as np


def rgb2gray(image):
    """
    Convert RGB image to grayscale

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        grayscale image as numpy array of shape (H, W)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    return image
