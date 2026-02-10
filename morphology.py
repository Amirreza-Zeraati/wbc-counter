import os
import cv2
import numpy as np
import processing.utilities as filters


def morphology_effects(image_path, output_folder="morph_comparison"):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    channel = filters.extract_channels(image, channel='s')
    median = cv2.medianBlur(channel, 5)
    _, binary = cv2.threshold(median, 90, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{output_folder}/00_binary_before_morph.jpg", binary)
    print("✓ Binary image (before morphology) saved")
    configs = [
        (3, 1, "3x3_iter1"),  # Small kernel, 1 iteration
        (3, 2, "3x3_iter2"),  # Small kernel, 2 iterations (current)
        (5, 2, "5x5_iter2"),  # Medium kernel
        (7, 2, "7x7_iter2"),  # Large kernel
        (3, 5, "3x3_iter5"),  # Many iterations
    ]

    for kernel_size, iterations, label in configs:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        cv2.imwrite(f"{output_folder}/open_{label}.jpg", opened)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        cv2.imwrite(f"{output_folder}/close_{label}.jpg", closed)
        diff_open = cv2.absdiff(binary, opened)
        diff_close = cv2.absdiff(opened, closed)
        cv2.imwrite(f"{output_folder}/diff_open_{label}.jpg", diff_open)
        cv2.imwrite(f"{output_folder}/diff_close_{label}.jpg", diff_close)
        print(f"✓ Tested: Kernel {kernel_size}x{kernel_size}, Iterations {iterations}")
    print(f"\nAll results saved in: {output_folder}/")
    print("Check 'diff_' images to see what each filter removed/added!")


if __name__ == "__main__":
    morphology_effects("dataset/Im024_1.jpg")
