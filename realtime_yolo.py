import time
import serial
import requests
import numpy as np
from io import BytesIO
from pprint import pprint

import cv2
import os
import random
from collections import Counter
import datetime

from requests.auth import HTTPBasicAuth
import json

# ==============================
# Parameters
# ==============================
FREEZE_FRAMES = 30  # YOLO 결과를 몇 프레임 동안 정지할지 설정

# Initialize serial communication with Arduino
ser = serial.Serial("/dev/ttyACM1", 9600)

# Configuration for YOLO API
ACCESS_KEY = "ezeJWt9iFMaP7HGvwYgds6Za1Sb35fwHaPZF89mi"
AUTH_USERNAME = "kdt2024_1-27"
START_INDEX = 0
END_INDEX = 300
IMAGE_EXTENSION = ".jpg"
headers = {"Content-Type": "image/jpg"}
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/8f81f503-b7c6-4220-8ad3-9e54ff2729c7/inference"

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

color_map = {}

def get_color_for_class(cls, color_map):
    """Assign a unique color to each class."""
    if cls not in color_map:
        # Assign a color from COLOR_LIST or generate a random one if exhausted
        if len(color_map) < len(COLOR_LIST):
            color_map[cls] = COLOR_LIST[len(color_map)]
        else:
            color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[cls]

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

        # Put text
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        y += line_height  # Move to next line

        # Check if y exceeds image height to prevent drawing outside the image
        if y > image.shape[0]:
            break  # Stop drawing if we reach the bottom of the image

def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img

def inference_request(img: np.array, api_url: str):
    """Send image to inference API endpoint and get the result."""
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    try:
        response = requests.post(
            url=api_url,
            auth=HTTPBasicAuth(AUTH_USERNAME, ACCESS_KEY),
            headers=headers,
            data=img_bytes,  # Send raw binary data
        )
        if response.status_code == 200:
            print("Image sent successfully")
            return response.json()
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
            print(f"Response content: {response.text}")  # debugging
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

# ==========================================================
# Main Loop
# ==========================================================

# 카메라를 한 번 열고 지속적으로 사용
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera Error")
    exit(-1)

freeze_count = 0
annotated_image = None

# crop_info가 필요하다면 설정
crop_info = {"x": 870, "y": 110, "width": 600, "height": 530}

try:
    while True:
        # 시리얼 데이터 확인(비차단으로 처리)
        if ser.in_waiting > 0:
            data = ser.read()
            print(f"Received data: {data}")
            if data == b"0":
                # data==b"0"일 때 현재 프레임으로 YOLO 처리
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture image")
                    continue

                # 필요하면 크롭
                if crop_info is not None:
                    frame = crop_img(frame, crop_info)

                # 원본 이미지 저장
                original_folder = 'original'
                if not os.path.exists(original_folder):
                    os.makedirs(original_folder)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                original_image_path = os.path.join(original_folder, f"{timestamp}.jpg")
                cv2.imwrite(original_image_path, frame)
                print(f"Saved original image to {original_image_path}")

                # YOLO inference
                result = inference_request(frame, api_url)
                if result is not None:
                    objects = result.get('objects', [])
                    if not objects:
                        print("No objects detected.")
                        # 객체 미검출 시에도 정지 프레임 유지 가능. 필요 없으면 건너뛰기
                        annotated_image = frame.copy()
                    else:
                        print(f"Number of objects detected: {len(objects)}")
                        label_counts = Counter(obj.get('class', 'N/A') for obj in objects)

                        annotated_image = draw_bounding_boxes(frame.copy(), objects, color_map)
                        draw_label_counts(annotated_image, label_counts, color_map)

                        # YOLO 폴더에 결과 저장
                        yolo_folder = 'Yolo'
                        if not os.path.exists(yolo_folder):
                            os.makedirs(yolo_folder)
                        annotated_image_path = os.path.join(yolo_folder, f"{timestamp}_annotated.jpg")
                        cv2.imwrite(annotated_image_path, annotated_image)
                        print(f"Annotated image saved to {annotated_image_path}")
                else:
                    print("Failed to get inference result.")
                    annotated_image = frame.copy()  # 실패 시에도 freeze 상태에서는 현재 프레임 그대로 보여줌

                # YOLO 결과를 FREEZE_FRAMES 동안 정지
                freeze_count = FREEZE_FRAMES

        # 매 루프마다 카메라 프레임 읽기
        ret, live_frame = cam.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        if crop_info is not None:
            live_frame = crop_img(live_frame, crop_info)

        if freeze_count > 0:
            # YOLO 결과 정지 상태
            cv2.imshow('Live', annotated_image)
            freeze_count -= 1
            if freeze_count == 0:
                # 정지 상태 해제 후, Arduino에 '1' 전송
                ser.write(b"1")
        else:
            # 실시간 프레임 표시
            cv2.imshow('Live', live_frame)

        # 키입력 (q: 종료)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
