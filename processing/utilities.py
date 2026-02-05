import numpy as np
from collections import deque


def extract_channels(img, channel='s'):
    bgr = img.astype(np.float32) / 255.0
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]

    c_max = np.max(bgr, axis=2)
    c_min = np.min(bgr, axis=2)
    delta = c_max - c_min

    if channel == 'h':
        h = np.zeros_like(c_max)
        mask_delta = delta > 0
        mask_r = (c_max == red) & mask_delta
        h[mask_r] = (green[mask_r] - blue[mask_r]) / delta[mask_r]
        h[mask_r] = np.mod(h[mask_r], 6)

        mask_g = (c_max == green) & mask_delta
        h[mask_g] = ((blue[mask_g] - red[mask_g]) / delta[mask_g]) + 2
        mask_b = (c_max == blue) & mask_delta
        h[mask_b] = ((red[mask_b] - green[mask_b]) / delta[mask_b]) + 4

        h = h * 60.0
        h[h < 0] += 360
        h_final = (h / 2).astype(np.uint8)
        return h_final

    elif channel == 's':
        s = np.zeros_like(c_max)
        non_zero_v = c_max > 0
        s[non_zero_v] = delta[non_zero_v] / c_max[non_zero_v]
        s_final = (s * 255).astype(np.uint8)
        return s_final

    elif channel == 'v':
        v_final = (c_max * 255).astype(np.uint8)
        return v_final

    else:
        print("select correct channel (h-s-v)")
        return None


def median_blur(img, kernel_size=3):
    pad_size = kernel_size // 2
    rows, cols = img.shape
    padded_img = np.pad(img, (pad_size, pad_size), mode='edge')
    shifted_layers = []

    for r_shift in range(-pad_size, pad_size + 1):
        for c_shift in range(-pad_size, pad_size + 1):
            r_start = pad_size + r_shift
            r_end = r_start + rows
            c_start = pad_size + c_shift
            c_end = c_start + cols
            layer = padded_img[r_start:r_end, c_start:c_end]
            shifted_layers.append(layer)

    stack = np.dstack(shifted_layers)
    output_img = np.median(stack, axis=2).astype(np.uint8)
    return output_img


def threshold(img, threshold_value=127):
    binary_img = np.zeros_like(img)
    binary_img[img > threshold_value] = 255
    return binary_img


def count_objects_manual(binary_img, min_area=50):
    rows, cols = binary_img.shape
    visited = np.zeros((rows, cols), dtype=bool)

    wbc_count = 0
    component_stats = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] == 255 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                current_area = 0

                while q:
                    curr_r, curr_c = q.popleft()
                    current_area += 1
                    for dr, dc in neighbors:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary_img[nr, nc] == 255 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                if current_area >= min_area:
                    wbc_count += 1
                    component_stats.append(current_area)
    return wbc_count, component_stats


def filter_and_count_manual(binary_img, min_area=80):
    """
    1. Finds connected components (WBC candidates).
    2. Calculates their area.
    3. If area > min_area: Keeps them in 'final_mask' and counts them.
    4. If area <= min_area: Ignores them (effectively removing noise).
    """
    rows, cols = binary_img.shape
    visited = np.zeros((rows, cols), dtype=bool)
    final_mask = np.zeros((rows, cols), dtype=np.uint8)  # The "Clean" image

    count = 0

    # Standard BFS directions (8-connectivity)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] == 255 and not visited[r, c]:
                # --- New Component Found ---
                component_pixels = []  # Store coordinates to "draw" later
                q = deque([(r, c)])
                visited[r, c] = True
                component_pixels.append((r, c))

                # Run BFS to find the whole object
                while q:
                    curr_r, curr_c = q.popleft()

                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc

                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary_img[nr, nc] == 255 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                component_pixels.append((nr, nc))

                # --- Filter Logic (The replacement for contourArea > 80) ---
                area = len(component_pixels)

                if area > min_area:
                    count += 1
                    # "Draw" the object onto the final mask
                    # This replaces cv2.drawContours(final_mask, ...)
                    for (pr, pc) in component_pixels:
                        final_mask[pr, pc] = 255

    return count, final_mask

# Usage in your pipeline:
# count, clean_binary = filter_and_count_manual(noisy_binary, min_area=80)

