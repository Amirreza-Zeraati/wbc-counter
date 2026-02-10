import os
import cv2
import numpy as np
import processing.utilities as filters


def visualize_pipeline_steps(image_path, output_folder="pipeline_steps"):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    print(f"Processing: {image_path}")
    print(f"Saving steps to: {output_folder}/\n")
    cv2.imwrite(f"{output_folder}/00_original.jpg", image)
    print("✓ Step 0: Original image saved")
    channel = filters.extract_channels(image, channel='s')
    cv2.imwrite(f"{output_folder}/01_s_channel.jpg", channel)
    print("✓ Step 1: S-Channel extraction (HSV Saturation)")
    median = cv2.medianBlur(channel, 5)
    cv2.imwrite(f"{output_folder}/02_median_blur.jpg", median)
    print("✓ Step 2: Median Blur (noise removal)")
    _, binary = cv2.threshold(median, 90, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{output_folder}/03_binary_threshold.jpg", binary)
    print("✓ Step 3: Binary Threshold (black & white)")
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(f"{output_folder}/04_morphology_open.jpg", opened)
    print("✓ Step 4: Morphological Opening (remove small noise)")
    mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(f"{output_folder}/05_morphology_close.jpg", mask)
    print("✓ Step 5: Morphological Closing (fill holes)")
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(f"{output_folder}/06_gradient.jpg", gradient)
    print("✓ Step 6: Morphological Gradient (edge detection)")
    centers = cv2.subtract(mask, gradient)
    cv2.imwrite(f"{output_folder}/07_centers_subtracted.jpg", centers)
    print("✓ Step 7: Subtract Gradient (isolate cell centers)")
    dist_transform = cv2.distanceTransform(centers, cv2.DIST_L2, 5)
    dist_visual = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"{output_folder}/08_distance_transform.jpg", dist_visual)
    print("✓ Step 8: Distance Transform (create peaks)")
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.imwrite(f"{output_folder}/09_sure_foreground.jpg", sure_fg)
    print("✓ Step 9: Aggressive Threshold (find peak tips - 70%)")
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    cv2.imwrite(f"{output_folder}/10_sure_background.jpg", sure_bg)
    print("✓ Step 10: Sure Background (dilated mask)")
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite(f"{output_folder}/11_unknown_region.jpg", unknown)
    print("✓ Step 11: Unknown Region (area between cells)")

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_visual = np.uint8(markers * (255 / (ret + 1)))
    markers_colored = cv2.applyColorMap(markers_visual, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_folder}/12_markers.jpg", markers_colored)
    print(f"✓ Step 12: Markers (found {ret} distinct regions)")

    markers_watershed = cv2.watershed(image, markers.copy())
    image_with_boundaries = image.copy()
    image_with_boundaries[markers_watershed == -1] = [0, 0, 255]  # Red boundaries
    cv2.imwrite(f"{output_folder}/13_watershed_boundaries.jpg", image_with_boundaries)
    print("✓ Step 13: Watershed Algorithm (separate touching cells)")

    final_mask = np.zeros_like(channel, dtype=np.uint8)
    count = 0

    for marker_id in range(2, ret + 1):
        cell_mask = np.zeros_like(channel, dtype=np.uint8)
        cell_mask[markers_watershed == marker_id] = 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 80:
            count += 1
            cv2.drawContours(final_mask, [c], -1, 255, -1)

    cv2.imwrite(f"{output_folder}/14_final_mask_before_bfs.jpg", final_mask)
    print(f"✓ Step 14: Final Mask (extracted {count} regions)")

    count_bfs, clean_mask = filters.filter_and_count_manual(final_mask, min_area=80)
    cv2.imwrite(f"{output_folder}/15_bfs_counted_final.jpg", clean_mask)
    print(f"✓ Step 15: BFS Counting (verified count = {count_bfs} cells)")

    clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([image, clean_mask_bgr])
    cv2.imwrite(f"{output_folder}/16_comparison_original_vs_result.jpg", comparison)
    print("✓ Step 16: Side-by-side comparison saved")

    print("\n" + "=" * 50)
    print(f"Final Count: {count_bfs} White Blood Cells")
    print(f"All steps saved in: {output_folder}/")
    print("=" * 50)


if __name__ == "__main__":
    image_path = "dataset/Im024_1.jpg"
    visualize_pipeline_steps(image_path, output_folder="debug_steps")
