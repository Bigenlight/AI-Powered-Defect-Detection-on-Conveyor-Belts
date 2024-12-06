import cv2
import requests
from requests.auth import HTTPBasicAuth
import json
import os
import random
from collections import Counter

# -------------------- Configuration --------------------
INPUT_FOLDER = "/home/theo/Downloads/after_sharpened_2"
OUTPUT_FOLDER = "/home/theo/Downloads/result_YoloV6/"
URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/3600b672-a324-48ce-bb00-20a462c1a8ab/inference"
ACCESS_KEY = "ezeJWt9iFMaP7HGvwYgds6Za1Sb35fwHaPZF89mi"
AUTH_USERNAME = "kdt2024_1-27"
IMAGE_EXTENSION = ".jpg"  # Supported image extension
# -------------------------------------------------------

# Define a list of distinct colors (BGR format)
COLOR_LIST = [
    (255, 0, 0),      # Blue
    (50, 205, 0),     # Green
    (0, 0, 255),      # Red
    (225, 205, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (50, 0, 128),     # Navy
    (128, 128, 0),    # Olive
    # Add more colors as needed
]

def get_color_for_class(cls, color_map):
    """Assign a unique color to each class."""
    if cls not in color_map:
        # Assign a color from COLOR_LIST or generate a random one if exhausted
        if len(color_map) < len(COLOR_LIST):
            color_map[cls] = COLOR_LIST[len(color_map)]
        else:
            color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[cls]

def load_image(image_path):
    """Load image from the specified path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

def send_image_for_detection(image_path):
    """Send image to YOLO endpoint and get detection results."""
    with open(image_path, 'rb') as f:
        image_data = f.read()

    headers = {"Content-Type": "image/jpg"}

    try:
        response = requests.post(
            url=URL,
            auth=HTTPBasicAuth(AUTH_USERNAME, ACCESS_KEY),
            headers=headers,
            data=image_data,
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Request failed: {e}")

    if response.status_code != 200:
        raise ConnectionError(f"Request failed with status code {response.status_code}: {response.text}")

    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON")

    return result

def draw_bounding_boxes(image, objects, color_map):
    """Draw bounding boxes and labels on the image with class-specific colors."""
    for obj in objects:
        cls = obj.get('class', 'N/A')
        score = obj.get('score', 0)
        box = obj.get('box', [])

        if len(box) != 4:
            print(f"Invalid box format for object {obj}")
            continue

        x1, y1, x2, y2 = box

        # Validate coordinates are integers
        try:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        except ValueError:
            print(f"Non-integer box coordinates for object {obj}")
            continue

        # Get color for the current class
        color = get_color_for_class(cls, color_map)

        # Draw rectangle with the class-specific color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{cls}: {score:.2f}"

        # Choose font and get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Calculate label background coordinates
        label_bg_x1 = x1
        label_bg_y1 = y1 - text_height - baseline if y1 - text_height - baseline > 0 else y1 + text_height + baseline
        label_bg_x2 = x1 + text_width
        label_bg_y2 = y1

        # Ensure label background is within image boundaries
        label_bg_y1 = max(label_bg_y1, 0)
        label_bg_x2 = min(label_bg_x2, image.shape[1])

        # Draw filled rectangle for label background with the same class color
        cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, thickness=cv2.FILLED)

        # Determine text color based on background brightness for readability
        brightness = sum(color)
        text_color = (0, 0, 0) if brightness > 600 else (255, 255, 255)  # Black or white

        # Put text on the label background
        cv2.putText(image, label, (label_bg_x1, label_bg_y2 - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image

def draw_label_counts(image, label_counts, color_map):
    """Draw label counts in the top-left corner of the image."""
    # Set starting position
    x = 10
    y = 20  # Initial y position

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Calculate line height based on font size
    (text_width, text_height), baseline = cv2.getTextSize('Text', font, font_scale, thickness)
    line_height = text_height + baseline + 5  # Adjust as needed

    for label, count in label_counts.items():
        text = f"{label}: {count}"
        color = get_color_for_class(label, color_map)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw rectangle background for text
        cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), color, cv2.FILLED)

        # Determine text color based on background brightness
        brightness = sum(color)
        text_color = (0, 0, 0) if brightness > 600 else (255, 255, 255)

        # Put text on the image
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        y += line_height  # Move to next line

        # Check if y exceeds image height to prevent drawing outside the image
        if y > image.shape[0]:
            break  # Stop drawing if we reach the bottom of the image

def is_perfect_image(label_counts):
    """Determine if the image meets the criteria for a 'perfect' image."""
    required_counts = {
        'BOOTSEL': 1,
        'USB': 1,
        'CHIPSET': 1,
        'OSCILLATOR': 1,
        'RASPBERRY PICO': 1,
        'HOLE': 4
    }
    for label, count in required_counts.items():
        if label_counts.get(label, 0) != count:
            return False
    return True

def main():
    print("Starting batch processing...")

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
    else:
        print(f"Output directory already exists: {OUTPUT_FOLDER}")

    # Create 'perfect' and 'defective' subfolders
    perfect_folder = os.path.join(OUTPUT_FOLDER, 'perfect')
    defective_folder = os.path.join(OUTPUT_FOLDER, 'defective')

    for folder in [perfect_folder, defective_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

    # List all files in the input directory with the specified extension
    all_files = os.listdir(INPUT_FOLDER)
    image_files = [file for file in all_files if file.lower().endswith(IMAGE_EXTENSION.lower())]

    if not image_files:
        print(f"No image files with extension '{IMAGE_EXTENSION}' found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(image_files)} image(s) to process.")

    # Initialize a global color map for consistent coloring across all images
    global_color_map = {}

    for image_filename in image_files:
        input_image_path = os.path.join(INPUT_FOLDER, image_filename)
        # Extract base name and extension
        base_name, ext = os.path.splitext(image_filename)
        output_image_filename = f"{base_name}_annotated{ext}"

        print(f"\nProcessing Image: {image_filename}")

        try:
            # Load the original image
            image = load_image(input_image_path)

            # Send image to YOLO for detection
            detection_result = send_image_for_detection(input_image_path)

            # Extract detected objects
            objects = detection_result.get('objects', [])
            if not objects:
                print("No objects detected. Saving the original image.")
                # Decide output folder
                output_subfolder = defective_folder  # No detections, consider defective
                output_image_path = os.path.join(output_subfolder, output_image_filename)
                cv2.imwrite(output_image_path, image)
                print(f"Saved image without annotations to {output_image_path}")
                continue

            print(f"Number of objects detected: {len(objects)}")

            # Count the occurrences of each label
            label_counts = Counter(obj.get('class', 'N/A') for obj in objects)

            # Determine if the image is 'perfect' or 'defective'
            is_perfect = is_perfect_image(label_counts)
            if is_perfect:
                output_subfolder = perfect_folder
                print("Image classified as PERFECT.")
            else:
                output_subfolder = defective_folder
                print("Image classified as DEFECTIVE.")

            output_image_path = os.path.join(output_subfolder, output_image_filename)

            # Draw bounding boxes and labels
            annotated_image = draw_bounding_boxes(image, objects, global_color_map)

            # Draw label counts in the top-left corner
            draw_label_counts(annotated_image, label_counts, global_color_map)

            # Save the annotated image
            cv2.imwrite(output_image_path, annotated_image)
            print(f"Annotated image saved to {output_image_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_filename}: {e}")
            continue

    print("\nBatch processing completed.")

if __name__ == "__main__":
    main()
