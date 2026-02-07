import os
import cv2
import numpy as np
import processing.utilities as filters


def process_single_image(image_path, output_folder, index):
    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # --- Pipeline Step 1: Preprocessing ---
    # Extract Saturation Channel
    channel = filters.extract_channels(image, channel='s')

    # Median Blur
    median = cv2.medianBlur(channel, 5)

    # Thresholding
    _, binary = cv2.threshold(median, 90, 255, cv2.THRESH_BINARY)

    # Morphological Operations (Noise Removal)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Pipeline Step 2: Method 3 (Morphological Gradient Separation) ---
    # Morphological gradient highlights boundaries
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    # Subtract gradient from mask to emphasize centers
    centers = cv2.subtract(mask, gradient)

    # Distance transform
    dist_transform = cv2.distanceTransform(centers, cv2.DIST_L2, 5)

    # Aggressive threshold (0.7) for separating touching cells
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)

    # Extract cells
    final_mask = np.zeros_like(channel)
    count = 0

    for marker_id in range(2, ret + 1):
        cell_mask = np.zeros_like(channel)
        cell_mask[markers == marker_id] = 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 80:
                count += 1
                cv2.drawContours(final_mask, [c], -1, 255, -1)

    # --- Create Output Image (Side-by-Side) ---
    # Convert final mask to BGR so we can stack it with the original image
    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    # Stack original image and processed mask horizontally
    combined_image = np.hstack([image, final_mask_bgr])

    # Construct filename: index_count.jpg (e.g., 1_23.jpg)
    filename = f"{index}_{count}.jpg"
    output_path = os.path.join(output_folder, filename)

    # Save result
    cv2.imwrite(output_path, combined_image)
    print(f"Processed: {filename} (Count: {count})")


def batch_process(input_folder, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files from input folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    files.sort()  # Sort to process in consistent order

    if not files:
        print(f"No image files found in '{input_folder}'")
        return

    print(f"Found {len(files)} images in '{input_folder}'. Processing...")

    for i, filename in enumerate(files, start=1):
        image_path = os.path.join(input_folder, filename)
        process_single_image(image_path, output_folder, i)

    print(f"\nDone! All results saved in '{output_folder}'")


if __name__ == "__main__":
    # Define your folders here
    INPUT_FOLDER = "dataset"  # Folder containing your original images
    OUTPUT_FOLDER = "batch_results"  # Folder to save results

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)