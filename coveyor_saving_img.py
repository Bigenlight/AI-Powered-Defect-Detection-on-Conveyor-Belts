import serial
import cv2
import os
import datetime

# Initialize serial communication with Arduino
ser = serial.Serial("/dev/ttyACM0", 9600)

def get_img():
    """Capture an image from the USB camera.

    Returns:
        numpy.array: Captured image as a NumPy array.
    """
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    if not ret:
        print("Failed to capture image")
        return None

    return img

def crop_img(img, size_dict):
    """Crop the image based on the provided coordinates.

    Args:
        img (numpy.array): The original image.
        size_dict (dict): Dictionary containing 'x', 'y', 'width', and 'height' for cropping.

    Returns:
        numpy.array: Cropped image.
    """
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    cropped_img = img[y : y + h, x : x + w]
    return cropped_img

def save_image(img, folder, timestamp):
    """Save the image to the specified folder with a timestamp.

    Args:
        img (numpy.array): The image to save.
        folder (str): The folder path where the image will be saved.
        timestamp (str): Timestamp string to include in the filename.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

    image_path = os.path.join(folder, f"{timestamp}.jpg")
    cv2.imwrite(image_path, img)
    print(f"Saved image to {image_path}")

while True:
    try:
        data = ser.read()
        print(f"Received data: {data}")
        if data == b"0":
            img = get_img()
            if img is None:
                print("No image captured. Skipping...")
                ser.write(b"1")  # Resume conveyor belt even if no image was captured
                continue

            # Optional cropping (uncomment and adjust if needed)
            crop_info = {"x": 870, "y": 110, "width": 520, "height": 520}
            if crop_info:
                img = crop_img(img, crop_info)
                print("Image cropped.")

            # Save the image into 'original' folder
            original_folder = 'original'
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_image(img, original_folder, timestamp)

            # Send '1' back to Arduino to resume conveyor belt
            ser.write(b"1")
            print("Sent '1' to Arduino to resume conveyor belt.")
        else:
            # Optional: Handle other data if needed
            pass
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        continue
