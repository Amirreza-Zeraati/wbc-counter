import cv2
import numpy as np
import processing.utilities as filters


def detect_and_count(image):
    # 1. Extract Saturation channel (Pure Python implementation in utilities)
    channel = filters.extract_channels(image, channel='s')
    
    # 2. Apply Median Blur (Pure Python implementation in utilities)
    # Using kernel_size=5 as in your original code
    median = filters.median_blur(channel, kernel_size=5)
    
    # 3. Thresholding (Pure Python implementation in utilities)
    # Threshold > 90 to separate cells from background
    binary = filters.threshold(median, threshold_value=90)

    # 4. Morphological Operations (Noise Removal)
    # Keeping OpenCV here for performance/robustness, as requested "as much as I can"
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional: Use Distance Transform + Threshold to separate touching cells
    # This creates a "sure foreground" mask where cells are separated
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Threshold at 50% of max distance
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Count using BFS (Pure Python implementation in utilities)
    # We pass 'sure_fg' which contains the separated cell blobs
    count, final_mask = filters.filter_and_count_manual(sure_fg, min_area=80)

    return final_mask, count
