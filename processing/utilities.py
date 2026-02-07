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


def filter_and_count_manual(binary_img, min_area=80):
    rows, cols = binary_img.shape
    visited = np.zeros((rows, cols), dtype=bool)
    final_mask = np.zeros((rows, cols), dtype=np.uint8)
    count = 0

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] == 255 and not visited[r, c]:
                component_pixels = []
                q = deque([(r, c)])
                visited[r, c] = True
                component_pixels.append((r, c))
                while q:
                    curr_r, curr_c = q.popleft()
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary_img[nr, nc] == 255 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                component_pixels.append((nr, nc))
                area = len(component_pixels)
                if area > min_area:
                    count += 1
                    for (pr, pc) in component_pixels:
                        final_mask[pr, pc] = 255
    return count, final_mask
