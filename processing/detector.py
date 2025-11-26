import cv2
import numpy as np


def detect_and_count(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(image)
    enhanced = 255 - g

    ret, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    final_mask = np.zeros_like(gray)
    count = 0

    for marker_id in range(2, ret + 1):
        cell_mask = np.zeros_like(gray)
        cell_mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 80:
                count += 1
                cv2.drawContours(final_mask, [c], -1, 255, -1)

    return final_mask, count
