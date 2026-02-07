import cv2
import numpy as np
import processing.utilities as filters


def _segment_method3(image, min_contour_area=80):
    channel = filters.extract_channels(image, channel='s')
    median = cv2.medianBlur(channel, 5)
    _, binary = cv2.threshold(median, 90, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    centers = cv2.subtract(mask, gradient)

    dist_transform = cv2.distanceTransform(centers, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    final_mask = np.zeros_like(channel, dtype=np.uint8)

    for marker_id in range(2, ret + 1):
        cell_mask = np.zeros_like(channel, dtype=np.uint8)
        cell_mask[markers == marker_id] = 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > min_contour_area:
            cv2.drawContours(final_mask, [c], -1, 255, -1)
    return final_mask


def detect_and_count(image):
    final_mask = _segment_method3(image)
    count, clean_mask = filters.filter_and_count_manual(final_mask, min_area=80)
    return clean_mask, int(count)
