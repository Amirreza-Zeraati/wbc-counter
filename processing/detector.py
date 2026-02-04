import cv2
import numpy as np
import processing.utilities as filters


def detect_and_count(image):
    # 1. Extract Saturation channel (Pure Python implementation in utilities)
    channel = filters.extract_channels(image, channel='s')
    
    # 2. Apply Median Blur (Pure Python implementation in utilities)
    median = filters.median_blur(channel, kernel_size=5)
    
    # 3. Thresholding (Pure Python implementation in utilities)
    binary = filters.threshold(median, threshold_value=90)

    # 4. Morphological Operations (Noise Removal)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Separate touching/adherent cells using distance transform
    # Distance transform: each pixel value = distance to nearest background pixel
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # More aggressive thresholding to find cell centers (peaks in distance map)
    # Use 60% instead of 50% to create better separated markers
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. Find cell centers and their approximate radii using BFS
    cell_centers, cell_radii = filters.find_cell_centers_and_radii(
        sure_fg, 
        dist_transform, 
        min_area=80
    )
    
    # 7. Create final mask with circular cells drawn
    final_mask = np.zeros_like(channel)
    count = len(cell_centers)
    
    # Draw circular cells on the final mask
    for (cy, cx), radius in zip(cell_centers, cell_radii):
        filters.draw_filled_circle(final_mask, cy, cx, radius)

    return final_mask, count
