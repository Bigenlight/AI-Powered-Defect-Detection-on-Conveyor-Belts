import cv2
import numpy as np
import os

def sharpen_image(image):
    """
    Apply a sharpening filter to the given image.

    Args:
        image (numpy.ndarray): Original image.

    Returns:
        numpy.ndarray: Sharpened image.
    """
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def process_folder(input_folder, output_folder):
    """
    Apply sharpening filter to all images in the input folder and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save sharpened images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_path):
            try:
                image = cv2.imread(input_path)

                if image is None:
                    print(f"Skipping invalid image file: {input_path}")
                    continue

                sharpened_image = sharpen_image(image)

                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{base_name}_sharpened{ext}")

                success = cv2.imwrite(output_path, sharpened_image)

                if success:
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Failed to save: {output_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_folder = "/home/rokey/Desktop/sharpenit/before"  # Replace with your input folder absolute path
    output_folder = "/home/rokey/Desktop/sharpenit/after"  # Replace with your output folder absolute path

    process_folder(input_folder, output_folder)
