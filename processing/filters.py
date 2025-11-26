import cv2
import numpy as np

def apply_clahe(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

def apply_morphology(binary_mask):
    kernel = np.ones((3, 3), np.uint8)

    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return clean_mask


def apply_watershed(binary_mask):
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  
    markers[unknown == 255] = 0

   
    num_cells = ret - 1 
    return num_cells, markers
import cv2

def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
